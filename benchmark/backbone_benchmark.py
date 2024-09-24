"""
This module contains the high level functions for benchmarking on a single node.
"""
import importlib
from functools import partial
from typing import Any

import mlflow
import optuna
import pandas as pd
import torch
from jsonargparse import CLI
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from optuna.pruners import HyperbandPruner
from tabulate import tabulate

from benchmark.benchmark_types import (
    Defaults,
    ParameterBounds,
    Task,
    combine_with_defaults,
    optimization_space_type,
)
from benchmark.model_fitting import fit_model, fit_model_with_hparams

direction_type_to_optuna = {"min": "minimize", "max": "maximize"}


def benchmark_backbone_on_task(
    defaults: Defaults,
    task: Task,
    storage_uri: str,
    experiment_name: str,
    optimization_space: optimization_space_type | None = None,
    n_trials: int = 1,
    save_models: bool = False,
    precision: _PRECISION_INPUT = "16-mixed",
) -> tuple[float, str | list[str] | None, dict[str, Any]]:
    with mlflow.start_run(
        run_name=task.name,
        nested=True,
    ) as run:
        training_spec = combine_with_defaults(task, defaults)
        task = training_spec.task
        lightning_task_class = training_spec.task.type.get_class_from_enum()

        # if no optimization params, just run it
        if optimization_space is None:
            return (
                *fit_model(
                    training_spec,
                    lightning_task_class,
                    f"{run.info.run_name}",
                    experiment_name,
                    storage_uri,
                    run.info.run_id,
                    save_models=save_models,
                ),
                {},
            )

        # if optimization parameters specified, do hyperparameter tuning
        study = optuna.create_study(
            direction=direction_type_to_optuna[
                training_spec.task.direction
            ],  # in the future may want to allow user to specify this
            pruner=HyperbandPruner(),
        )
        objective = partial(
            fit_model_with_hparams,
            training_spec,
            lightning_task_class,
            task.name,
            experiment_name,
            optimization_space,
            storage_uri,
            run.info.run_id,
            save_models,
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            # callbacks=[champion_callback],
            catch=[torch.cuda.OutOfMemoryError],  # add a few more here?
        )
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric(f"best_{task.metric}", study.best_value)
        return study.best_value, task.metric, study.best_trial.params


# Custom function to parse the optimization space argument
def parse_optimization_space(space: dict | None) -> optimization_space_type | None:
    if space is None:
        return None
    parsed_space: optimization_space_type = {}
    for key, value in space.items():
        if isinstance(value, dict):
            try:
                bounds = ParameterBounds(**value)
                parsed_space[key] = bounds
            except TypeError:
                # Recursively parse nested optimization spaces
                parsed_space[key] = parse_optimization_space(value)
        elif isinstance(value, list):
            # If it's a list, leave it as is
            parsed_space[key] = value
        else:
            raise ValueError(f"Invalid type for {key}: {value}")
    return parsed_space

def benchmark_backbone(
    defaults: Defaults,
    tasks: list[Task],
    experiment_name: str,
    storage_uri: str,
    ray_storage_path: str | None = None,
    backbone_import: str | None = None,
    run_name: str | None = None,
    n_trials: int = 1,
    optimization_space: dict | None = None,
    save_models: bool = False,
    run_id: str | None = None,
):
    """Highest level function to benchmark a backbone using a single node

    Args:
        defaults (Defaults): Defaults that are set for all tasks
        tasks (list[Task]): List of Tasks to benchmark over. Will be combined with defaults to get the final parameters of the task.
        experiment_name (str): Name of the MLFlow experiment to be used.
        storage_uri (str): Path to storage location.
        ray_storage_path (str | None): Ignored. Exists for compatibility with ray configs.
        backbone_import (str | None): Path to module that will be imported to register a potential new backbone. Defaults to None.
        run_name (str | None, optional): Name of highest level mlflow run. Defaults to None.
        n_trials (int, optional): Number of hyperparameter optimization trials to run. Defaults to 1.
        optimization_space (optimization_space_type | None, optional): Parameters to optimize over. Should be a dictionary
            of strings (parameter name) to list (discrete set of possibilities) or ParameterBounds, defining a range to optimize over.
            Arguments belonging passed to the backbone, decoder or head should be given in the form `backbone_{argument}`, `decoder_{argument}` or `head_{argument}` Defaults to None.
        save_models (bool, optional): Whether to save the model. Defaults to False.
        run_id (str | None): id of existing mlflow run to use as top-level run. Useful to add more experiments to a previous benchmark run. Defaults to None.
        precision (str): precision to use for training. Defaults to 16-mixed.

    """
    if backbone_import:
        importlib.import_module(backbone_import)
    mlflow.set_tracking_uri(storage_uri)
    mlflow.set_experiment(experiment_name)

    optimization_space = parse_optimization_space(optimization_space)
    table_columns = ["Task", "Metric", "Best Score", "Hyperparameters"]
    table_entries = []
    with mlflow.start_run(run_name=run_name, run_id=run_id) as run:
        mlflow.set_tag("purpose", "backbone_benchmarking")
        for task in tasks:
            best_value, metric_name, hparams = benchmark_backbone_on_task(
                defaults,
                task,
                storage_uri,
                experiment_name,
                optimization_space=optimization_space,
                n_trials=n_trials,
                save_models=save_models,
            )
            table_entries.append([task.name, metric_name, best_value, hparams])

        table = tabulate(table_entries, headers=table_columns)
        print(table)
        df = pd.DataFrame(data=table_entries, columns=table_columns)
        df.set_index("Task")
        mlflow.log_table(
            df,
            "results_table.json",
            run.info.run_id,
        )

def main():
    CLI(benchmark_backbone, fail_untyped=False)


if __name__ == "__main__":
    main()
