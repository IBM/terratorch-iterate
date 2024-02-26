from functools import partial
from typing import Any

import mlflow
import optuna
import pandas as pd
import torch
from jsonargparse import CLI
from tabulate import tabulate

from benchmark.model_fitting import fit_model, fit_model_with_hparams
from benchmark.types import (
    Backbone,
    Task,
    TaskTypeEnum,
    optimization_space_type,
)

EXPERIMENT_NAME = "backbone_benchmark"


# override Optuna's default logging to ERROR only
optuna.logging.set_verbosity(optuna.logging.ERROR)


# define a logging callback that will report on only new challenger parameter configurations if a
# trial has usurped the state of 'best conditions'
def champion_callback(study: optuna.Study, frozen_trial):
    """
    From: https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.html
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """
    if len(study.trials) == 0:
        return
    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (
                abs(winner - study.best_value) / study.best_value
            ) * 100
            print(
                "=" * 40
                + "\n"
                + f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement" + "\n" + "=" * 40 + "\n"
            )
        else:
            print(
                "=" * 40
                + "\n"
                + f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}"
                + "\n"
                + "=" * 40
                + "\n"
            )


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
) -> tuple[float, str | list[str] | None, dict[str, Any]]:
    with mlflow.start_run(
        run_name=f"{backbone.backbone}_{task.name}", nested=True
    ) as run:
        lightning_task_class = task.type.get_class_from_enum()
        model_args = build_model_args(backbone, task)

        # if no optimization params, just run it
        if optimization_space is None:
            return (
                *fit_model(
                    backbone,
                    model_args,
                    task,
                    lightning_task_class,
                    f"{run.info.run_name}",
                    storage_uri,
                    EXPERIMENT_NAME,
                    save_models=save_models,
                ),
                {},
            )

        # if optimization parameters specified, do hyperparameter tuning
        study = optuna.create_study(
            direction="minimize"  # in the future may want to allow user to specify this
        )
        objective = partial(
            fit_model_with_hparams,
            backbone,
            task,
            lightning_task_class,
            model_args,
            f"{backbone.backbone}_{task.name}",
            optimization_space,
            storage_uri,
            EXPERIMENT_NAME,
            save_models,
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            # callbacks=[champion_callback],
            catch=[torch.cuda.OutOfMemoryError],  # add a few more here?
        )

        return study.best_value, task.metric, study.best_trial.params


def benchmark_backbone(
    backbone: Backbone,
    tasks: list[Task],
    storage_uri: str,
    benchmark_suffix: str | None = None,
    n_trials: int = 1,
    optimization_space: optimization_space_type | None = None,
    save_models: bool = True,
):
    mlflow.set_tracking_uri(storage_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.pytorch.autolog(log_datasets=False)
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


def main():
    CLI(benchmark_backbone, fail_untyped=False)


if __name__ == "__main__":
    main()
