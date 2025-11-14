from inspect_ai import task, Task, eval
from data_utils.return_dataset import return_dataset

from data_utils.load_mcqa_task import load_mcqa_dataset
from utils.cache import Cache, CacheType
from utils.setup import load_config
from utils.enums import get_scorers_for_metrics, Metrics
from model_utils.irt import PyMCIRTModel
import os


@task
def mcqa_metrics(sample_to_score: dict = None, metrics_config: dict = None, base_config: dict = None):
    """Task that loads a dataset and runs metrics with all specified parameters."""
    
    # Use passed configs (they come through task_args)
    if metrics_config is None or base_config is None:
        raise ValueError("metrics_config and base_config must be provided through task_args")
    
    cfg_local = metrics_config
    global_cfg_local = base_config
    
    metric_list = global_cfg_local.get("scoring_metrics", [])
    if not metric_list:
        raise ValueError("You must specify at least one metric")
    Metrics.validate_metrics_list(metric_list)
    
    # Handle difficulty metric with IRT
    if Metrics.DIFFICULTY.value in metric_list and sample_to_score == None:
        difficulty_scorers = get_scorers_for_metrics([Metrics.DIFFICULTY.value], cfg_local)
        scorers = difficulty_scorers[0]
    else:
        all_scorers = get_scorers_for_metrics(metric_list, cfg_local, sample_to_score=sample_to_score)
        scorers = []
        for metric_scorers in all_scorers:
            if isinstance(metric_scorers, list):
                scorers.extend(metric_scorers)
            else:
                scorers.append(metric_scorers)
    
    dataset_path = global_cfg_local.get("dataset")
    dataset = load_mcqa_dataset(dataset_path)
    return Task(dataset=dataset, solver=return_dataset(), scorer=scorers)


def run_metrics_eval(metrics_cfg=None, base_cfg=None):
    """Run metrics evaluation using eval() function.
    
    Args:
        metrics_cfg: Metrics configuration dictionary
        base_cfg: Base configuration dictionary
    """
    if metrics_cfg is None or base_cfg is None:
        raise ValueError("metrics_cfg and base_cfg must be provided")
    
    # Initialize cache
    cache = Cache(cache_dir=base_cfg.get("cache_dir"), cache_type=CacheType(base_cfg.get('cache_type', 'none')))
    
    metric_list = base_cfg.get("scoring_metrics", [])
    if not metric_list:
        raise ValueError("You must specify at least one metric")
    Metrics.validate_metrics_list(metric_list)
    
    # Resolve run_name from CLI -> global config
    resolved_run_name = base_cfg.get("metric_run_name")

    # get the name of the model (doesn't really matter but Inspect needs it, so will pick the first valid model)
    for metric in metric_list:
        model = metrics_cfg.get(metric, {}).get("models", [None])[0]
        if model is None:
            model = metrics_cfg.get(metric, {}).get("model", None)
        if model:
            break

    sample_to_score = dict()

    # Check if difficulty metric is specified
    if Metrics.DIFFICULTY.value in metric_list:

        # 0a) Ensure we have the right models
        if metrics_cfg.get("difficulty", {}).get("run_irt", True): 
            difficulty_cfg = metrics_cfg.get("difficulty", {})
            model_abilities = PyMCIRTModel.load_fixed_abilities(base_cfg.get("cache_dir") + f"/irt/{base_cfg.get('skill_run_name')}/", name='model_skills')
            print(set([model.replace('/', '_') for model in difficulty_cfg.get("models", [])]).difference(set(model_abilities.keys())))
            assert set([model.replace('/', '_') for model in difficulty_cfg.get("models", [])]).difference(set(model_abilities.keys())) == set(), "You do not have IRT abilities for the models you specified"

        # 0b) Run initial evaluation to get accuracy scores
        eval_logs_acc = cache.load('eval_logs', resolved_run_name, 'data_eval_logs_acc')
        if eval_logs_acc == None:
            os.environ["INSPECT_EVAL_LOG_FILE_PATTERN"] = f"{resolved_run_name}_mcqa_data_accuracy"
            eval_logs_acc = eval(
                "endpoints/run_metrics.py@mcqa_metrics",
                model=model,
                limit=metrics_cfg.get("num_samples", None),
                task_args={
                    "metrics_config": metrics_cfg,
                    "base_config": base_cfg,
                },
            )
            cache.save('eval_logs', resolved_run_name, 'data_eval_logs_acc', eval_logs_acc)

        # 0c) Run IRT with cached model abilities to get difficulty/discriminability scores if specified
        if metrics_cfg.get("difficulty", {}).get("run_irt", True):
            sample_to_score = cache.load('irt_logs', resolved_run_name, 'data_sample_to_score')
            if sample_to_score == None:
                irt_model = PyMCIRTModel(eval_logs_acc)
                irt_model.train(
                    fixed_abilities=model_abilities,
                    draws=metrics_cfg.get("difficulty", {}).get("irt_model", {}).get("num_draws", 200),
                    tune=metrics_cfg.get("difficulty", {}).get("irt_model", {}).get("num_tune", 200),
                    chains=metrics_cfg.get("difficulty", {}).get("irt_model", {}).get("chains", 5),
                    cores=metrics_cfg.get("difficulty", {}).get("irt_model", {}).get("cores", 5),
                )
                sample_to_score = irt_model.save(base_cfg.get("cache_dir", {}) + '/irt/' + f'/{resolved_run_name}/',
                                                base_cfg.get("plot_dir", {}) + '/irt/' + f'/{resolved_run_name}/',
                                                'dataset_irt')
                cache.save('irt_logs', resolved_run_name, 'data_sample_to_score', sample_to_score)

    # 1) run normal evaluation
    report_card_logs = cache.load('eval_logs', resolved_run_name, 'report_card_logs')
    if report_card_logs == None:
        os.environ["INSPECT_EVAL_LOG_FILE_PATTERN"] = f"{resolved_run_name}_mcqa_data_report_card"
        report_card_logs = eval(
            "endpoints/run_metrics.py@mcqa_metrics",
            model=model,
            limit=metrics_cfg.get("num_samples", None),
            task_args={
                "sample_to_score": sample_to_score,
                "metrics_config": metrics_cfg,
                "base_config": base_cfg,
            },
        )
        cache.save('eval_logs', resolved_run_name, 'report_card_logs', report_card_logs)