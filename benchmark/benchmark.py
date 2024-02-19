import enum
import sys
from dataclasses import dataclass, field
from typing import Any

import albumentations
import mlflow
import terratorch
import torch
from jsonargparse import CLI
from lightning import LightningDataModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
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
    max_epochs: int = 100
    freeze_backbone: bool = False
    decoder_args: dict[str, Any] = field(default_factory=dict)
    head_args: dict[str, Any] = field(default_factory=dict)
    ignore_index: int | None = None


def build_model_args(backbone: Backbone, task: Task):
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
):
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        trainer.logger = MLFlowLogger(
            experiment_name=EXPERIMENT_NAME,
            run_id=run.info.run_id,
            save_dir="/dccstor/geofm-finetuning/carlosgomes/benchmark",
            log_model=True,
        )
        trainer.fit(task, datamodule=datamodule)
        trainer.test(task, datamodule=datamodule)


def benchmark_backbone_on_task(backbone: Backbone, task: Task):
    with mlflow.start_run(
        run_name=f"{backbone.backbone_name}_{task.name}", nested=True
    ) as run:
        lightning_task_class = get_class_from_enum(task.type)
        model_args = build_model_args(backbone, task)

        lrs = [1e-4, 1e-3, 1e-2, 1e-1]
        for run_number, lr in enumerate(lrs):
            lightning_task = lightning_task_class(
                model_args,
                backbone.model_factory,
                loss=task.loss,
                lr=lr,
                optimizer=torch.optim.AdamW,
                freeze_backbone=task.freeze_backbone,
                ignore_index=task.ignore_index,
            )
            trainer = Trainer(
                callbacks=[
                    RichProgressBar(),
                    EarlyStopping(monitor="val/loss", patience=20),
                ],
                max_epochs=task.max_epochs,
            )
            launch_training(
                trainer,
                lightning_task,
                task.datamodule,
                f"{run.info.run_name}_{run_number}",
            )

        trainer.fit(lightning_task, datamodule=task.datamodule)
        trainer.test(lightning_task, datamodule=task.datamodule)


def benchmark_backbone(
    backbone: Backbone, tasks: list[Task], benchmark_suffix: str | None = None
):
    mlflow.set_tracking_uri("/dccstor/geofm-finetuning/carlosgomes/benchmark")
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = backbone.backbone_name
    if benchmark_suffix:
        run_name += f"_{benchmark_suffix}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("purpose", "backbone_benchmarking")
        for task in tasks:
            benchmark_backbone_on_task(backbone, task)


def main():
    CLI(benchmark_backbone)


if __name__ == "__main__":
    main()
