import os

from data_utils.refine_dataset import DifficultyRefineType

from inspect_ai.dataset import Dataset
from inspect_ai.log import EvalLog
import json
import numpy as np
from typing import Any, List, Dict, Optional


import pymc as pm
import matplotlib.pyplot as plt

class PyMCIRTModel:
    """PyMC-based IRT model with flexible parameter fixing capabilities."""
    
    def __init__(self, eval_logs: List[EvalLog]):
        self.trace = None
        self.model = None
        self.sample_to_score = {}
        self.responses = self._convert_eval_logs_to_responses(eval_logs)
        
    
    def _convert_eval_logs_to_responses(self, eval_logs: list[EvalLog]):
        """Convert eval_logs to response format and store original metadata."""
        responses = []
        self.sample_to_score = {}
        
        for eval_log in eval_logs:
            for eval_sample in eval_log.samples:
                for score_name, score_value in eval_sample.scores.items():
                    responses.append({
                        "question_id": f"q{eval_sample.id}",
                        "model_id": score_name[len('accuracy_'):],
                        "accuracy": int(score_value.value)
                    })
                    if eval_sample.id not in self.sample_to_score:
                        self.sample_to_score[eval_sample.id] = {}
                    if 'accuracy' not in self.sample_to_score[eval_sample.id]:
                        self.sample_to_score[eval_sample.id]['accuracy'] = {}
                    self.sample_to_score[eval_sample.id]['accuracy'][score_name[len('accuracy_'):]] = {'score': score_value.value, 'answer': score_value.answer, 'explanation': score_value.explanation}
        return responses
        
    def train(self, 
              fixed_abilities: Optional[Dict[str, float]] = None,
              draws: int = 500,
              tune: int = 500,
              chains: int = 3,
              cores: int = 3) -> None:
        """Train PyMC IRT model with optional fixed abilities."""

        self.chains = chains
        self.fixed_abilities = fixed_abilities

        # Convert responses to arrays
        question_ids = [r["question_id"] for r in self.responses]
        model_ids = [r["model_id"] for r in self.responses]
        accuracies = np.array([r["accuracy"] for r in self.responses])
        
        # Create mappings
        unique_questions = list(set(question_ids))
        unique_models = list(set(model_ids))
        
        question_to_idx = {q: i for i, q in enumerate(unique_questions)}
        model_to_idx = {m: i for i, m in enumerate(unique_models)}
        idx_to_model = {i: m for m, i in model_to_idx.items()}
        
        # Convert to indices
        question_indices = np.array([question_to_idx[q] for q in question_ids])
        model_indices = np.array([model_to_idx[m] for m in model_ids])
        
        N_questions = len(unique_questions)
        N_models = len(unique_models)
        
        # Store mappings for later use
        self.question_to_idx = question_to_idx
        self.model_to_idx = model_to_idx
        self.unique_questions = unique_questions
        self.unique_models = unique_models
        
        with pm.Model() as model:
            # Item parameters (difficulty and discriminability)
            difficulty = pm.Normal("difficulty", mu=0, sigma=1, shape=N_questions)
            discriminability = pm.LogNormal("discriminability", mu=0, sigma=0.5, shape=N_questions)
            
            # Model abilities - PyMC 5.x approach
            if fixed_abilities is not None:
                ability = np.array([fixed_abilities[idx_to_model[model_idx]] for model_idx in range(N_models)])
            else:
                # Learn abilities from data
                ability = pm.Normal("ability", mu=0, sigma=1, shape=N_models)
            
            # IRT model: P(correct) = sigmoid(discriminability * (ability - difficulty))
            logits = discriminability[question_indices] * (ability[model_indices] - difficulty[question_indices])
            p_correct = pm.Deterministic("p_correct", pm.math.sigmoid(logits))
            
            # Likelihood
            accuracy_obs = pm.Bernoulli("accuracy", p=p_correct, observed=accuracies)
            
            # Sample from posterior using JAX backend (NumPyro) to avoid C compilation
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                init="advi",
                random_seed=42,
                progressbar=True,
                discard_tuned_samples=False,
                idata_kwargs={"log_likelihood": True},
                return_inferencedata=True,
                target_accept=0.9
            )
            self.model = model
    
    def get_item_parameters(self) -> Dict[str, Dict[str, float]]:
        """Extract item difficulty and discriminability parameters."""
        if self.trace is None:
            raise ValueError("Model must be trained before extracting parameters")
        
        # Get posterior means
        difficulty_mean = self.trace.posterior["difficulty"].mean(dim=["chain", "draw"]).values
        discriminability_mean = self.trace.posterior["discriminability"].mean(dim=["chain", "draw"]).values
        
        # Convert to dictionary format
        item_params = {}
        for i, question_id in enumerate(self.unique_questions):
            item_params[question_id] = {
                "difficulty": float(difficulty_mean[i]),
                "discriminability": float(discriminability_mean[i])
            }
        
        return item_params
    
    def calculate_fisher_information(self, ability_values: Optional[List[float]] = None) -> Dict[str, Dict[str, float]]:
        """Calculate Fisher information for each item."""
        if self.trace is None:
            raise ValueError("Model must be trained before calculating Fisher information")
        
        difficulty_mean = self.trace.posterior["difficulty"].mean(dim=["chain", "draw"]).values
        discriminability_mean = self.trace.posterior["discriminability"].mean(dim=["chain", "draw"]).values
        
        if ability_values is None:
            if self.fixed_abilities is not None:
                ability_values = list(self.fixed_abilities.values())
            else:
                ability_mean = self.trace.posterior["ability"].mean(dim=["chain", "draw"]).values
                ability_values = ability_mean.tolist()
        
        fisher_info_results = {}
        
        for i, question_id in enumerate(self.unique_questions):
            difficulty = difficulty_mean[i]
            discriminability = discriminability_mean[i]
            
            fisher_info_values = []
            for ability in ability_values:
                # Fisher information: I(θ) = a² * p(θ) * (1 - p(θ))
                p_correct = 1 / (1 + np.exp(-discriminability * (ability - difficulty)))
                fisher_info = discriminability**2 * p_correct * (1 - p_correct)
                fisher_info_values.append(fisher_info)
            
            mean_fisher_info = np.mean(fisher_info_values)
            max_fisher_info = np.max(fisher_info_values)
            max_fisher_idx = np.argmax(fisher_info_values)
            optimal_ability = ability_values[max_fisher_idx]
            
            fisher_info_results[question_id] = {
                "mean_fisher_information": float(mean_fisher_info),
                "max_fisher_information": float(max_fisher_info),
                "optimal_ability": float(optimal_ability),
                "fisher_information_at_abilities": {
                    str(ability): float(fi) for ability, fi in zip(ability_values, fisher_info_values)
                }
            }
        
        return fisher_info_results
    
    def get_model_abilities(self) -> Dict[str, float]:
        """Extract model ability parameters."""
        if self.trace is None:
            raise ValueError("Model must be trained before extracting parameters")
        
        if self.fixed_abilities:
            assert "ability" not in self.trace.posterior, "You have fixed abilities but the model was trained with learnable abilities"
            return self.fixed_abilities

        # Get posterior means
        ability_mean = self.trace.posterior["ability"].mean(dim=["chain", "draw"]).values
        
        # Convert to dictionary format
        model_abilities = {}
        for i, model_id in enumerate(self.unique_models):
            model_abilities[model_id] = float(ability_mean[i])
        
        return model_abilities
    
    def save(self, save_path: str, plot_path: str, name: str, include_fisher_info: bool = True) -> Dict[int, Dict[str, float]]:
        """Save model parameters to JSON file and return sample_to_score."""
        if self.trace is None:
            raise ValueError("Model must be trained before saving")
        
        # Extract parameters
        item_params = self.get_item_parameters()
        model_abilities = self.get_model_abilities()
        
        # Create save dictionary
        save_data = {
            "item_parameters": item_params,
            "model_abilities": model_abilities
        }
        
        # Add Fisher information if requested
        if include_fisher_info:
            fisher_info = self.calculate_fisher_information()
            save_data["fisher_information"] = fisher_info
        
        # Save to file
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, name + '_params.json'), 'w') as f:
            json.dump(save_data, f, indent=2)
        
        # Generate and save training plot
        self.plot_training(plot_path, name)
        
        return self.get_sample_to_score()
    
    def plot_training(self, path: str, name: str) -> None:
        """Create and save training plot showing log likelihood for accuracy over time."""

        plt.figure(figsize=(6, 3))

        # Average across chains
        lp_chains = []
        for chain in range(self.chains):
            lp_total = np.concatenate((self.trace.warmup_sample_stats["lp"][chain], self.trace.sample_stats["lp"][chain]))
            lp_chains.append(lp_total)
        
        lp_avg = np.mean(lp_chains, axis=0)
        plt.plot(lp_avg)

        plt.title("Log Probabilities Over Time")
        plt.xlabel("Sample")
        plt.ylabel("Log Probability")
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(path, f'{name}_training_plot.pdf')
        os.makedirs(path, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def load_fixed_abilities(path: str, name: str) -> Dict[str, float]:
        """Load fixed abilities from cached parameters file."""
        with open(os.path.join(path, name + '_params.json'), 'r') as f:
            cached_params = json.load(f)  
        return cached_params['model_abilities']
    
    def get_sample_to_score(self, include_fisher_info: bool = True) -> Dict[int, Dict]:
        """Add IRT parameters to existing sample_to_score metadata."""
        if self.trace is None:
            raise ValueError("Model must be trained before extracting parameters")
        
        item_params = self.get_item_parameters()
        
        # Add IRT parameters to existing metadata
        for question_id, params in item_params.items():
            sample_id = int(question_id[1:])
            if sample_id in self.sample_to_score:
                self.sample_to_score[sample_id]['difficulty'] = params['difficulty']
                self.sample_to_score[sample_id]['discriminability'] = params['discriminability']
        
        # Add Fisher information if requested
        if include_fisher_info:
            fisher_info = self.calculate_fisher_information()
            for question_id, fi_data in fisher_info.items():
                sample_id = int(question_id[1:])
                if sample_id in self.sample_to_score:
                    self.sample_to_score[sample_id]['mean_fisher_information'] = fi_data['mean_fisher_information']
                    self.sample_to_score[sample_id]['max_fisher_information'] = fi_data['max_fisher_information']
                    self.sample_to_score[sample_id]['optimal_ability'] = fi_data['optimal_ability']
        
        return self.sample_to_score

def _apply_min_discrimination_filter(samples: List[Dict], item_params: Dict[str, Dict[str, float]], min_discrimination: float) -> List[Dict]:
    """Filter out samples with discriminability below minimum threshold."""
    filtered_samples = []
    for sample_info in samples:
        sample_id = sample_info['sample_id']
        if sample_id in item_params:
            discriminability = item_params[sample_id].get('discriminability', 0.0)
            if discriminability >= min_discrimination:
                filtered_samples.append(sample_info)
        else:
            # If no IRT params available, keep the sample
            filtered_samples.append(sample_info)
    return filtered_samples

def _resolve_max_samples(filter_config: Dict[str, Any], filter_type: DifficultyRefineType, total_samples: int) -> int:
    """Resolve the number of samples to keep based on max_size configuration."""

    type_config = filter_config.get(filter_type.value, {})
    max_size = type_config.get("max_size", filter_config.get("max_size", None))

    if max_size is None:
        return total_samples

    if isinstance(max_size, float):
        if not 0.0 <= max_size <= 1.0:
            raise ValueError("max_size must be between 0.0 and 1.0 when specified as a float")
        resolved = int(total_samples * max_size)
    elif isinstance(max_size, int):
        if max_size <= 0:
            raise ValueError("max_size must be greater than 0 when specified as an integer")
        resolved = max_size
    else:
        raise ValueError("max_size must be either an integer or a float")

    return max(1, min(total_samples, resolved))


def filter_dataset_by_irt(dataset: Dataset, item_params: Dict[str, Dict[str, float]], filter_type: DifficultyRefineType, filter_config: Dict[str, Any], sample_to_score: Optional[Dict[int, Dict]] = None, min_discrimination: Optional[float] = None) -> Dict[str, Dataset]:
    """
    Filter dataset based on IRT parameters.
    
    Args:
        dataset: The dataset to filter
        item_params: Dictionary mapping sample IDs to IRT parameters (difficulty, discriminability)
        filter_type: Type of filtering to apply
        filter_config: Configuration for the filtering (e.g., max_size limits)
        sample_to_score: Optional sample metadata containing Fisher information and other IRT data
        min_discrimination: Optional minimum discriminability threshold to filter out low-quality items
    
    Returns:
        Dictionary mapping filter level names to filtered datasets
    """
    from inspect_ai.dataset import MemoryDataset, Sample
    
    # Convert dataset to list of samples for easier manipulation
    samples = []
    for i, sample in enumerate(dataset):
        # Use a unique ID that won't conflict
        sample_id = i + 1
        samples.append({
            'sample': sample,
            'sample_id': sample_id,
            'index': i
        })
    
    # Apply minimum discrimination filter if specified
    if min_discrimination is not None:
        samples = _apply_min_discrimination_filter(samples, item_params, min_discrimination)
    
    if filter_type == DifficultyRefineType.NONE:
        return {"all": dataset}
    
    if filter_type == DifficultyRefineType.SATURATION:
        # Sort samples by difficulty (descending - hardest first)
        samples_with_difficulty = []
        for sample_info in samples:
            sample_id = sample_info['sample_id']
            if f"q{sample_id}" in item_params:
                difficulty = item_params[f"q{sample_id}"].get('difficulty', 0.0)
                samples_with_difficulty.append((sample_info, difficulty))
            else:
                samples_with_difficulty.append((sample_info, 0.0))
        
        # Sort by difficulty (hardest first)
        samples_with_difficulty.sort(key=lambda x: x[1], reverse=True)

        total_samples = len(samples_with_difficulty)
        num_samples = _resolve_max_samples(filter_config, DifficultyRefineType.SATURATION, total_samples)

        hard_samples = []
        for item in samples_with_difficulty[:num_samples]:
            sample = item[0]['sample']
            sample_id = item[0]['sample_id']
            # Set the id field on the sample
            sample.id = sample_id
            hard_samples.append(sample)

        return {"hard": MemoryDataset(hard_samples), "all": dataset}
    
    elif filter_type == DifficultyRefineType.INFORMATIVE:
        # Sort samples by discriminability (descending - most discriminative first)
        samples_with_discriminability = []
        for sample_info in samples:
            sample_id = sample_info['sample_id']
            if f"q{sample_id}" in item_params:
                discriminability = item_params[f"q{sample_id}"].get('discriminability', 0.0)
                samples_with_discriminability.append((sample_info, discriminability))
            else:
                # If no IRT params available, assign neutral discriminability
                samples_with_discriminability.append((sample_info, 0.0))
        
        # Sort by discriminability (descending)
        samples_with_discriminability.sort(key=lambda x: x[1], reverse=True)
        
        total_samples = len(samples_with_discriminability)
        num_samples = _resolve_max_samples(filter_config, DifficultyRefineType.INFORMATIVE, total_samples)
        
        top_samples = []
        for item in samples_with_discriminability[:num_samples]:
            sample = item[0]['sample']
            sample_id = item[0]['sample_id']
            # Set the id field on the sample
            sample.id = sample_id
            top_samples.append(sample)
        
        return {"informative": MemoryDataset(top_samples), "all": dataset}
    
    elif filter_type == DifficultyRefineType.EFFICIENCY:
        if sample_to_score is None:
            raise ValueError("Sample metadata is required for efficiency filtering")
        
        samples_with_efficiency = []
        for sample_info in samples:
            sample_id = sample_info['sample_id']
            if sample_id in sample_to_score:
                mean_fi = sample_to_score[sample_id].get('mean_fisher_information', 0.0)
                samples_with_efficiency.append((sample_info, mean_fi))
            else:
                samples_with_efficiency.append((sample_info, 0.0))
        
        samples_with_efficiency.sort(key=lambda x: x[1], reverse=True)
        
        total_samples = len(samples_with_efficiency)
        num_samples = _resolve_max_samples(filter_config, DifficultyRefineType.EFFICIENCY, total_samples)
        
        top_samples = []
        for item in samples_with_efficiency[:num_samples]:
            sample = item[0]['sample']
            sample_id = item[0]['sample_id']
            # Set the id field on the sample
            sample.id = sample_id
            top_samples.append(sample)
        
        return {"efficient": MemoryDataset(top_samples), "all": dataset}
    
    elif filter_type == DifficultyRefineType.ADD_DISTRACTORS or filter_type == DifficultyRefineType.BLOOMS_TAXONOMY:
        # These are rewrite types, not filter types, so return the dataset unchanged
        # Rewriting will be handled elsewhere in the refinement pipeline
        return {"all": dataset}
    
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")