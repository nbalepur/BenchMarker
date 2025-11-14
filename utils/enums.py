from enum import Enum
from typing import Dict, List, Any
from inspect_ai.scorer import Scorer

from scorers.difficulty_scorer import difficulty_scorer, discriminability_scorer, get_accuracy_scorer_with_name, avg_accuracy_scorer
from scorers.shortcut_scorer import shortcut_scorer
from scorers.contamination_scorer import contamination_scorer
from scorers.writing_flaws_scorer import writing_flaws_scorer
from model_utils.web_search import WebSearchType


class Metrics(Enum):
    DIFFICULTY = "difficulty"
    SHORTCUTS = "shortcuts"
    CONTAMINATION = "contamination"
    WRITING_FLAWS = "writing_flaws"

    @classmethod
    def values(cls) -> list[str]:
        return [m.value for m in cls]
    
    @classmethod
    def validate_metrics_list(cls, metrics_list: list[str]) -> list[str]:
        """
        Validate that all metrics in the list are valid enum values.
        
        Args:
            metrics_list: List of metric strings to validate
            
        Returns:
            The validated metrics_list
            
        Raises:
            ValueError: If any metric in the list is not a valid enum value
        """
        if not isinstance(metrics_list, list):
            raise ValueError(f"metrics_list must be a list, got {type(metrics_list)}")
        
        valid_metrics = cls.values()
        invalid_metrics = []
        
        for metric in metrics_list:
            if metric not in valid_metrics:
                invalid_metrics.append(metric)
        
        if invalid_metrics:
            raise ValueError(
                f"Invalid metrics specified: {invalid_metrics}. "
                f"Valid metrics are: {valid_metrics}"
            )
        
        return metrics_list


def get_scorer_for_metric(metric: str, config: Dict[str, Any] | None = None, **kwargs) -> Scorer | List[Scorer]:
    """Get scorer for a metric with optional config parameters."""
    metric_config = config.get(metric, {})
    
    if metric == Metrics.DIFFICULTY.value:
        if kwargs.get('sample_to_score') == None:
            return [
                get_accuracy_scorer_with_name(name='accuracy_' + accuracy_model.replace('/', '_'))(
                model=accuracy_model,
                sample_to_score=kwargs.get('sample_to_score')
            ) for accuracy_model in metric_config["models"]]    
        else:    
            return [difficulty_scorer(sample_to_score=kwargs.get('sample_to_score')), discriminability_scorer(sample_to_score=kwargs.get('sample_to_score')), avg_accuracy_scorer(sample_to_score=kwargs.get('sample_to_score'))]

    elif metric == Metrics.SHORTCUTS.value:
        return shortcut_scorer(
            model=metric_config["model"],
            num_attempts=metric_config["attempts"],
            sample_to_score=kwargs.get('sample_to_score')
        )

    elif metric == Metrics.CONTAMINATION.value:
        search_type_str = metric_config.get("search_type", "perplexity")
        search_type = WebSearchType(search_type_str)
        return contamination_scorer(
            model=metric_config["model"],
            search_type=search_type,
            max_results=metric_config.get("max_results", 5),
            max_tokens_per_page=metric_config.get("max_tokens_per_page", 512),
            try_scraping=metric_config.get("try_scraping", False),
            attempts=metric_config.get("attempts", 3),
            sample_to_score=kwargs.get('sample_to_score')
        )

    elif metric == Metrics.WRITING_FLAWS.value:
        return writing_flaws_scorer(
            model=metric_config["model"],
            attempts=metric_config.get("attempts", 3),
            sample_to_score=kwargs.get('sample_to_score')
        )
        
    else:
        valid_metrics = [m.value for m in Metrics]
        raise KeyError(f"No scorer mapping for metric '{metric}'. Valid metrics: {valid_metrics}")

def get_scorers_for_metrics(metrics: List[str], config: Dict[str, Any] | None = None, **kwargs) -> List[Scorer]:
    return [get_scorer_for_metric(metric, config, **kwargs) for metric in metrics]