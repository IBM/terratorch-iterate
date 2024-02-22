import copy
import enum
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import albumentations  # noqa: F401
import mlflow
import optuna
import pandas as pd
import terratorch  # noqa: F401
import torch
from jsonargparse import CLI
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from terratorch.datasets import HLSBands
from terratorch.models import PrithviModelFactory
from terratorch.tasks import (
    IBMClassificationTask,
    IBMPixelwiseRegressionTask,
    IBMSemanticSegmentationTask,
)
from torchgeo.trainers import BaseTask

EXPERIMENT_NAME = "backbone_benchmark"


class TaskTypeEnum(enum.Enum):
    segmentation = "segmentation"
    regression = "regression"
    classification = "classification"


class ParameterTypeEnum(enum.Enum):
    integer = "int"
    real = "real"


def get_class_from_enum(
    task_type: TaskTypeEnum,
) -> type[
    IBMSemanticSegmentationTask | IBMClassificationTask | IBMPixelwiseRegressionTask
]:
    match task_type:
        case TaskTypeEnum.segmentation:
            return IBMSemanticSegmentationTask
        case TaskTypeEnum.regression:
            return IBMPixelwiseRegressionTask
        case TaskTypeEnum.classification:
            return IBMClassificationTask
        case _:
            raise TypeError("Task type does not exist")


@dataclass
class ParameterBounds:
    min: float | int
    max: float | int
    type: ParameterTypeEnum
    step: int | float | None = None


# jsonargparse does not seem to support recursive type defs, so support up to one level of nesting
optimization_space_type = dict[str, list | ParameterBounds]


@dataclass
class Backbone:
    backbone_name: str
    model_factory: str = "PrithviModelFactory"
    backbone_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    name: str
    type: TaskTypeEnum
    bands: list[HLSBands | int]
    datamodule: LightningDataModule
    decoder_name: str
    loss: str
    metric: str = "val/loss"
    lr: float = 1e-3
    max_epochs: int = 100
    freeze_backbone: bool = False
    decoder_args: dict[str, Any] = field(default_factory=dict)
    head_args: dict[str, Any] = field(default_factory=dict)
    ignore_index: int | None = None


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
    args["backbone"] = backbone.backbone_name
    for backbone_key, backbone_val in backbone.backbone_args.items():
        args[f"backbone_{backbone_key}"] = backbone_val
    args["pretrained"] = False

    args["decoder"] = task.decoder_name
    for decoder_key, decoder_val in task.decoder_args.items():
        args[f"decoder_{decoder_key}"] = decoder_val

    for head_key, head_val in task.head_args.items():
        args[f"head_{head_key}"] = head_val

    args["in_channels"] = len(task.bands)
    args["bands"] = task.bands

    if task.type != TaskTypeEnum.regression:
        args["num_classes"] = task.datamodule.num_classes

    return args


def launch_training(
    trainer: Trainer,
    task: BaseTask,
    datamodule: LightningDataModule,
    run_name: str,
    metric: str,
    storage_uri: str,
) -> float:
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        # explicitly log batch_size. Since it is not a model param, it will not be logged
        mlflow.log_param("batch_size", datamodule.batch_size)
        trainer.logger = MLFlowLogger(
            experiment_name=EXPERIMENT_NAME,
            run_id=run.info.run_id,
            save_dir=storage_uri,
            log_model=True,
        )
        trainer.fit(task, datamodule=datamodule)
        client = mlflow.tracking.MlflowClient(
            tracking_uri=storage_uri,
        )

        metric_history = client.get_metric_history(run.info.run_id, metric)
        if len(metric_history) == 0:
            raise Exception(
                f"No values for metric {metric}. Choose a valid metric for this task"
            )
        return metric_history[-1].value  # or best idk
        # trainer.test(task, datamodule=datamodule)


def inject_hparams(model_setup: dict[str, Any], model_hparams: dict[str, Any]):
    model_setup_with_injected_hparams = copy.deepcopy(model_setup)
    # assume maximum nesting value is 2
    for k, v in model_hparams.items():
        if k in model_setup_with_injected_hparams and isinstance(
            model_setup_with_injected_hparams[k], dict
        ):
            # overwrite / merge keys
            model_setup_with_injected_hparams[k] |= v
        else:
            # either add key or overwrite existing key
            model_setup_with_injected_hparams[k] = v
    return model_setup_with_injected_hparams


def fit_model(
    backbone: Backbone,
    model_args: dict,
    task: Task,
    lightning_task_class: type[BaseTask],
    run_name: str,
    storage_uri: str,
    lr: float | None = None,
    batch_size: int | None = None,
    save_models: bool = True,
) -> tuple[float, str]:
    if batch_size:
        task.datamodule.batch_size = (
            batch_size  # TODO: not sure if this will work, check
        )
    if lr is None:
        lr = task.lr
    lightning_task = lightning_task_class(
        model_args,
        backbone.model_factory,
        loss=task.loss,
        lr=lr,
        optimizer=torch.optim.AdamW,
        freeze_backbone=task.freeze_backbone,
        ignore_index=task.ignore_index,
        enable_checkpointing=save_models,
    )
    callbacks = [
        RichProgressBar(),
        EarlyStopping(monitor="val/loss", patience=5),  # let user configure this
    ]
    if save_models:
        callbacks.append(ModelCheckpoint(monitor="val/loss"))
    trainer = Trainer(
        callbacks=callbacks,
        max_epochs=task.max_epochs,
    )
    return launch_training(
        trainer, lightning_task, task.datamodule, run_name, task.metric, storage_uri
    ), task.metric


def fit_model_with_hparams(
    backbone: Backbone,
    task: Task,
    lightning_task_class: type[BaseTask],
    base_args: dict[str, Any],
    run_name: str,
    hparam_space: optimization_space_type,
    storage_uri: str,
    save_models: bool,
    trial: optuna.Trial,
) -> float:
    # treat lr and batch_size specially

    current_hparams: dict[str, int | float | str | bool] = {}

    for parameter, space in hparam_space.items():
        if isinstance(space, list):
            current_hparams[parameter] = trial.suggest_categorical(parameter, space)
        elif isinstance(space, ParameterBounds):
            match space.type:
                case ParameterTypeEnum.integer:
                    current_hparams[parameter] = trial.suggest_int(
                        parameter,
                        int(space.min),
                        int(space.max),
                        step=int(space.step) if space.step else 1,
                    )
                case ParameterTypeEnum.real:
                    current_hparams[parameter] = trial.suggest_float(
                        parameter, space.min, space.max, step=space.step
                    )
                case _:
                    raise Exception(
                        f"Type {space.type} not recognized. Suggest one of {[e.value for e in ParameterTypeEnum]}"
                    )
    lr = float(current_hparams.pop("lr", task.lr))
    batch_size = current_hparams.pop("batch_size", None)
    if batch_size is not None:
        batch_size = int(batch_size)
    model_args = inject_hparams(base_args, current_hparams)
    run_name = f"{run_name}_{trial.number}"
    return fit_model(
        backbone,
        model_args,
        task,
        lightning_task_class,
        run_name,
        storage_uri,
        lr=lr,
        batch_size=batch_size,
        save_models=save_models,
    )[0]  # return only the metric value for optuna


def benchmark_backbone_on_task(
    backbone: Backbone,
    task: Task,
    storage_uri: str,
    optimization_space: optimization_space_type | None = None,
    n_trials: int = 1,
    save_models: bool = True,
) -> tuple[float, str | list[str] | None, dict[str, Any]]:
    with mlflow.start_run(
        run_name=f"{backbone.backbone_name}_{task.name}", nested=True
    ) as run:
        lightning_task_class = get_class_from_enum(task.type)
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
                    save_models=save_models,
                ),
                {},
            )

        # if optimization parameters specified, do hyperparameter tuning
        study = optuna.create_study(
            direction="minimize"
        )  # in the future may want to allow user to specify this
        objective = partial(
            fit_model_with_hparams,
            backbone,
            task,
            lightning_task_class,
            model_args,
            f"{backbone.backbone_name}_{task.name}",
            optimization_space,
            storage_uri,
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
    run_name = backbone.backbone_name
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
    CLI(benchmark_backbone)


if __name__ == "__main__":
    main()
