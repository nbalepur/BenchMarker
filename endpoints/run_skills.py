from inspect_ai import task, Task, eval
from data_utils.return_dataset import return_dataset
from data_utils.merge_datasets import merge_and_shuffle_datasets
from utils.cache import Cache, CacheType
from utils.setup import load_config
from utils.enums import get_scorer_for_metric, Metrics
from model_utils.irt import PyMCIRTModel
from inspect_ai.scorer import Metric, SampleScore, Score, ScoreReducer, metric, score_reducer
import os


@task
def mcqa_skills(sample_to_score: dict = None, skills_config: dict = None, base_config: dict = None):
    """Task that takes multiple anchor datasets, shuffles them evenly, and runs evaluation to get model skills."""
    
    # Use passed configs (they come through task_args)
    if skills_config is None or base_config is None:
        raise ValueError("skills_config and base_config must be provided through task_args")
    
    cfg_local = skills_config
    global_cfg_local = base_config
    
    dataset_paths = cfg_local.get("skill_datasets", [])
    if not dataset_paths:
        raise ValueError("Datasets are required")
    
    merged_samples = merge_and_shuffle_datasets(dataset_paths)
    
    # Choose metric and parameters based on whether IRT is being used
    scorer = get_scorer_for_metric(Metrics.DIFFICULTY.value, cfg_local, sample_to_score=sample_to_score)
    return Task(
        dataset=merged_samples,
        solver=return_dataset(),
        scorer=scorer,
    )


def run_skills_eval(skills_cfg=None, base_cfg=None):
    """Run skills evaluation using eval() function.
    
    Args:
        skills_cfg: Skills configuration dictionary
        base_cfg: Base configuration dictionary
    """
    if skills_cfg is None or base_cfg is None:
        raise ValueError("skills_cfg and base_cfg must be provided")
    
    # Initialize cache
    cache = Cache(cache_dir=base_cfg.get("cache_dir"), cache_type=CacheType(base_cfg.get('cache_type', 'none')))
    
    model_list = skills_cfg.get("difficulty", {}).get("models", [])
    if not model_list:
        raise ValueError("Models are required")
    
    eval_model = model_list[0]

    # Resolve run_name from CLI -> skills config -> global config
    resolved_run_name = base_cfg.get("skill_run_name")
    
    # Get model accuracy
    skill_acc_eval_logs = cache.load('eval_logs', resolved_run_name, 'skill_acc_logs')
    if skill_acc_eval_logs == None:
        os.environ["INSPECT_EVAL_LOG_FILE_PATTERN"] = f"{resolved_run_name}_mcqa_skills"
        skill_acc_eval_logs = eval(
            "endpoints/run_skills.py@mcqa_skills",
            model=eval_model,
            limit=skills_cfg.get("num_samples", None),
            task_args={
                "skills_config": skills_cfg,
                "base_config": base_cfg,
            },
        )
        cache.save('eval_logs', resolved_run_name, 'skill_acc_logs', skill_acc_eval_logs)
    
    # 2) Run IRT
    sample_to_score = cache.load('irt_logs', resolved_run_name, 'skill_sample_to_score')
    if sample_to_score == None:
        irt_model = PyMCIRTModel(skill_acc_eval_logs)
        irt_model.train(
            draws=skills_cfg.get("difficulty", {}).get("irt_model", {}).get("num_draws", 200),
            tune=skills_cfg.get("difficulty", {}).get("irt_model", {}).get("num_tune", 200),
            chains=skills_cfg.get("difficulty", {}).get("irt_model", {}).get("chains", 3),
            cores=skills_cfg.get("difficulty", {}).get("irt_model", {}).get("cores", 3),
        )
        sample_to_score = irt_model.save(base_cfg.get("cache_dir", {}) + '/irt/' + f'/{resolved_run_name}/',
                                        base_cfg.get("plot_dir", {}) + '/irt/' + f'/{resolved_run_name}/',
                                        'model_skills')
        cache.save('irt_logs', resolved_run_name, 'skill_sample_to_score', sample_to_score)
    
    # 3) Add IRT scores
    skill_irt_eval_logs = cache.load('eval_logs', resolved_run_name, 'skill_irt_eval_logs')
    if skill_irt_eval_logs == None:
        os.environ["INSPECT_EVAL_LOG_FILE_PATTERN"] = f"{resolved_run_name}_mcqa_skills_with_irt"
        skill_irt_eval_logs = eval(
            "endpoints/run_skills.py@mcqa_skills",
            model=eval_model,
            limit=skills_cfg.get("num_samples", None),
            task_args={
                "sample_to_score": sample_to_score,
                "skills_config": skills_cfg,
                "base_config": base_cfg,
            },
        )
        cache.save('eval_logs', resolved_run_name, 'skill_irt_eval_logs', skill_irt_eval_logs)