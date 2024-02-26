from functools import partial
from typing import Any

import mlflow
import optuna
import pandas as pd
import ray
import torch
from jsonargparse import CLI
from ray import tune
from tabulate import tabulate

from benchmark.model_fitting import fit_model_with_hparams, ray_tune_model
from benchmark.types import (
    Backbone,
    Task,
    TaskTypeEnum,
    optimization_space_type,
)

EXPERIMENT_NAME = "ray_backbone_benchmark"


def build_model_args(backbone: Backbone, task: Task) -> dict[str, Any]:
    args = {}
    args["backbone"] = backbone.backbone
    for backbone_key, backbone_val in backbone.backbone_args.items():
        args[f"backbone_{backbone_key}"] = backbone_val

    # allow each task to specify / overwrite backbone keys
    for backbone_key, backbone_val in task.backbone_args.items():
        args[f"backbone_{backbone_key}"] = backbone_val
    args["pretrained"] = False

    args["decoder"] = task.decoder
    for decoder_key, decoder_val in task.decoder_args.items():
        args[f"decoder_{decoder_key}"] = decoder_val

    for head_key, head_val in task.head_args.items():
        args[f"head_{head_key}"] = head_val

    args["in_channels"] = len(task.bands)
    args["bands"] = task.bands

    if task.type != TaskTypeEnum.regression:
        if task.num_classes is not None:
            args["num_classes"] = task.num_classes
        else:
            if hasattr(task.datamodule, "num_classes"):
                args["num_classes"] = task.datamodule.num_classes
            elif hasattr(task.datamodule.dataset, "classes"):
                args["num_classes"] = len(task.datamodule.dataset.classes)
            else:
                raise Exception(
                    f"Could not infer num_classes. Please provide it explicitly for task {task.name}"
                )
    return args


def benchmark_backbone_on_task(
    backbone: Backbone,
    task: Task,
    storage_uri: str,
    optimization_space: optimization_space_type | None = None,
    n_trials: int = 1,
    save_models: bool = True,
) -> tuple[float, str | list[str] | None, dict[str, Any] | None]:
    with mlflow.start_run(
        run_name=f"{backbone.backbone}_{task.name}", nested=True
    ) as run:
        lightning_task_class = task.type.get_class_from_enum()
        model_args = build_model_args(backbone, task)

        # if no optimization params, just run it
        if optimization_space is None:
            raise Exception("For no optimiation space, run benchmark.py")

        results = ray_tune_model(
            backbone,
            task,
            lightning_task_class,
            model_args,
            f"{backbone.backbone}_{task.name}",
            optimization_space,
            storage_uri,
            EXPERIMENT_NAME,
            save_models,
            n_trials,
        )

        if results.get_best_result().metrics is None:
            raise Exception("Best result metrics were none")
        return (
            results.get_best_result().metrics[task.metric],
            task.metric,
            results.get_best_result().config,
        )


def benchmark_backbone(
    ray_address: str,
    backbone: Backbone,
    tasks: list[Task],
    storage_uri: str,
    benchmark_suffix: str | None = None,
    n_trials: int = 1,
    optimization_space: optimization_space_type | None = None,
    save_models: bool = True,
):
    ray.init(address=ray_address)
    mlflow.set_tracking_uri(storage_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    # mlflow.pytorch.autolog(log_datasets=False)
    run_name = backbone.backbone
    if benchmark_suffix:
        run_name += f"_{benchmark_suffix}"

    table_columns = ["Task", "Metric", "Best Score", "Hyperparameters"]
    table_entries = []
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("purpose", "backbone_benchmarking")
        for task in tasks:
            best_value, metric_name, hparams = benchmark_backbone_on_task(
                backbone,
                task,
                storage_uri,
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
        ray.shutdown()


def main():
    CLI(benchmark_backbone, fail_untyped=False)


if __name__ == "__main__":
    main()
