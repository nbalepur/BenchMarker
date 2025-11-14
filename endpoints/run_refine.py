from inspect_ai import task, Task, eval
from inspect_ai.dataset import Dataset
from data_utils.refine_dataset import MCQ, IRTFilterType, refine_dataset
from data_utils.return_dataset import return_dataset

from data_utils.load_mcqa_task import load_mcqa_dataset_from_logs
from utils.cache import Cache, CacheType
from utils.enums import get_scorers_for_metrics, Metrics
from model_utils.irt import PyMCIRTModel, filter_dataset_by_irt
from data_utils.save_annotations import save_refined_dataset
import pandas as pd
import os

@task
def refine_mcqa_dataset(sample_to_score: dict = None, refine_config: dict = None, base_config: dict = None, metrics_config: dict = None, report_card_logs: list[dict] = None, refined_dataset: Dataset = None):
    """Task that loads a dataset and runs metrics with all specified parameters."""
    
    # Use passed configs (they come through task_args)
    if refine_config is None or base_config is None:
        raise ValueError("metrics_config and base_config must be provided through task_args")
    
    metric_list = base_config.get("refining_metrics", [])
    if not metric_list:
        raise ValueError("You must specify at least one metric")
    Metrics.validate_metrics_list(metric_list)

    score_metrics_list = base_config.get("scoring_metrics", [])
    if score_metrics_list:
        Metrics.validate_metrics_list(score_metrics_list)

    if sample_to_score == None:
        scorers = []
        if Metrics.DIFFICULTY.value in score_metrics_list:
            difficulty_scorers = get_scorers_for_metrics([Metrics.DIFFICULTY.value], metrics_config)
            scorers = difficulty_scorers[0]
    else:
        all_scorers = get_scorers_for_metrics(score_metrics_list, metrics_config, sample_to_score=sample_to_score)
        scorers = []
        for metric_scorers in all_scorers:
            if isinstance(metric_scorers, list):
                scorers.extend(metric_scorers)
            else:
                scorers.append(metric_scorers)
    
    dataset = load_mcqa_dataset_from_logs(report_card_logs) if refined_dataset is None else refined_dataset

    solver = (
        return_dataset()
        if refined_dataset != None
        else refine_dataset(
            refine_config=refine_config,
            base_config=base_config,
            report_card_logs=report_card_logs,
        )
    )
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorers,
    )


def run_refine_eval(metrics_cfg=None, refine_cfg=None, base_cfg=None):
    """Run dataset refinement using eval() function.
    
    Args:
        metrics_cfg: Metrics configuration dictionary
        refine_cfg: Refine configuration dictionary
        base_cfg: Base configuration dictionary
    """
    if refine_cfg is None or base_cfg is None or metrics_cfg is None:
        raise ValueError("refine_cfg, base_cfg, and metrics_cfg must be provided")

    # 0a) If we need to run IRT, ensure we have the right models
    if Metrics.DIFFICULTY.value in base_cfg.get("scoring_metrics", []):
        difficulty_cfg = metrics_cfg.get("difficulty", {})
        model_abilities = PyMCIRTModel.load_fixed_abilities(base_cfg.get("cache_dir") + f"/irt/{base_cfg.get('skill_run_name')}/", name='model_skills')
        assert set([model.replace('/', '_') for model in difficulty_cfg.get("models", [])]).difference(set(model_abilities.keys())) == set(), "You do not have IRT abilities for the models you specified"
    
    # Initialize cache
    retr_cache = Cache(cache_dir=base_cfg.get("cache_dir"), cache_type=CacheType.CACHE)
    cache = Cache(cache_dir=base_cfg.get("cache_dir"), cache_type=CacheType(base_cfg.get('cache_type', 'none')))
    metric_run_name = base_cfg.get("metric_run_name")
    
    report_card_logs = retr_cache.load('eval_logs', metric_run_name, 'report_card_logs')
    if not report_card_logs:
        raise ValueError("You must run run_metrics before you can refine your dataset")

    resolved_run_name = base_cfg.get("metric_run_name")

    # Step 1: Refine the dataset
    os.environ["INSPECT_EVAL_LOG_FILE_PATTERN"] = f"{resolved_run_name}_refine_mcqa_dataset"
    refinement_logs = cache.load('eval_logs', resolved_run_name, 'refined_datasets')
    if refinement_logs == None:
        refinement_logs = eval(
            "endpoints/run_refine.py@refine_mcqa_dataset",
            model=refine_cfg.get('rewrite_model', 'openai/gpt-4o'),
            task_args={
                'sample_to_score': None,            
                'refine_config': refine_cfg,
                'base_config': base_cfg,
                'metrics_config': metrics_cfg,
                'report_card_logs': report_card_logs,
            },
        )
        cache.save('eval_logs', resolved_run_name, 'refined_datasets', refinement_logs)

    # Step 2: Save the refined dataset
    refined_dataset, original_sample_to_score = save_refined_dataset(refinement_logs, base_cfg, refine_cfg, report_card_logs, should_save=False)

    # Step 3: Run IRT if needed
    sample_to_score = original_sample_to_score.copy()  # Start with original cached results
    if Metrics.DIFFICULTY.value in base_cfg.get("scoring_metrics", []):
        refined_sample_to_score = cache.load('irt_logs', resolved_run_name, 'refined_sample_to_score')
        if refined_sample_to_score == None:
            irt_model = PyMCIRTModel(refinement_logs)
            irt_model.train(
                fixed_abilities=model_abilities,
                draws=metrics_cfg.get("irt_model", {}).get("num_draws", 200),
                tune=metrics_cfg.get("irt_model", {}).get("num_tune", 200),
                chains=metrics_cfg.get("irt_model", {}).get("chains", 3),
                cores=metrics_cfg.get("irt_model", {}).get("cores", 3),
            )
            refined_sample_to_score = irt_model.save(base_cfg.get("cache_dir", {}) + '/irt/' + f'/{resolved_run_name}/',
                                                    base_cfg.get("plot_dir", {}) + '/irt/' + f'/{resolved_run_name}/',
                                                    'refined_dataset_irt',
                                                    include_fisher_info=True)
            cache.save('irt_logs', resolved_run_name, 'refined_sample_to_score', refined_sample_to_score)
        
        # Merge original and refined results
        from data_utils.save_annotations import merge_sample_to_score
        sample_to_score = merge_sample_to_score(original_sample_to_score, refined_sample_to_score, refinement_logs)

    # Step 4: Run metrics on the refined dataset
    os.environ["INSPECT_EVAL_LOG_FILE_PATTERN"] = f"{resolved_run_name}_refined_mcqa_data_report_card"
    refined_mcqa_report_card_logs = cache.load('eval_logs', resolved_run_name, 'refined_mcqa_report_card_logs')
    if refined_mcqa_report_card_logs == None:
        refined_mcqa_report_card_logs = eval(
            "endpoints/run_refine.py@refine_mcqa_dataset",
            model=refine_cfg.get('rewrite_model', 'openai/gpt-4o'),
            task_args={
                'sample_to_score': sample_to_score,
                'refine_config': refine_cfg,
                'base_config': base_cfg,
                'metrics_config': metrics_cfg,
                'report_card_logs': report_card_logs,
                'refined_dataset': refined_dataset,
            },
        )
        cache.save('eval_logs', resolved_run_name, 'refined_mcqa_report_card_logs', report_card_logs)

    

    # Step 5: Do IRT-based filtering if needed
    if Metrics.DIFFICULTY.value in base_cfg.get("refining_metrics", []):

        if Metrics.DIFFICULTY.value not in base_cfg.get("scoring_metrics", []):
            print("You cannot use IRT-based filtering if you are not scoring the difficulty metric.")
            return

        # Build sample_to_score from refined report card logs so we can reuse cached metric results
        from data_utils.save_annotations import create_sample_to_score_from_refined_logs
        post_refine_sample_to_score = create_sample_to_score_from_refined_logs(
            refined_mcqa_report_card_logs,
            base_cfg,
        )

        irt_filter_type = IRTFilterType(refine_cfg.get("difficulty", {}).get("type", "none"))
        item_params = {
            sample.id: {
                'difficulty': sample.scores.get('diff', 0.0),
                'discriminability': sample.scores.get('disc', 0.0)
            }
            for sample in refined_mcqa_report_card_logs[0].samples
        }

        filtered_datasets = filter_dataset_by_irt(refined_dataset, item_params, irt_filter_type, refine_cfg.get("difficulty", {}), sample_to_score=refined_sample_to_score)

        for key_name, current_dataset in filtered_datasets.items():
            os.environ["INSPECT_EVAL_LOG_FILE_PATTERN"] = f"{resolved_run_name}_refined_mcqa_data_report_card-filter_irt_dataset={key_name}"
            post_irt_logs = cache.load('eval_logs', resolved_run_name, 'post_irt_logs')
            post_irt_logs = None

            if post_irt_logs == None:
                post_irt_logs = eval(
                    "endpoints/run_refine.py@refine_mcqa_dataset",
                    model=refine_cfg.get('rewrite_model', 'openai/gpt-4o'),
                    task_args={
                        'sample_to_score': post_refine_sample_to_score,
                        'refine_config': refine_cfg,
                        'base_config': base_cfg,
                        'metrics_config': metrics_cfg,
                        'report_card_logs': report_card_logs,
                        'refined_dataset': current_dataset,
                    },
                )
                cache.save('eval_logs', resolved_run_name, 'post_irt_logs', post_irt_logs)