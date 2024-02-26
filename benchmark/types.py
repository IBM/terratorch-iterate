import enum
from dataclasses import dataclass, field
from typing import Any

from terratorch.datasets import HLSBands
from terratorch.tasks import (
    IBMClassificationTask,
    IBMPixelwiseRegressionTask,
    IBMSemanticSegmentationTask,
)
from torchgeo.datamodules import BaseDataModule

valid_task_types = type[
    IBMSemanticSegmentationTask | IBMClassificationTask | IBMPixelwiseRegressionTask
]


class TaskTypeEnum(enum.Enum):
    segmentation = "segmentation"
    regression = "regression"
    classification = "classification"

    def get_class_from_enum(
        self,
    ) -> valid_task_types:
        match self:
            case TaskTypeEnum.segmentation:
                return IBMSemanticSegmentationTask
            case TaskTypeEnum.regression:
                return IBMPixelwiseRegressionTask
            case TaskTypeEnum.classification:
                return IBMClassificationTask
            case _:
                raise TypeError("Task type does not exist")


class ParameterTypeEnum(enum.Enum):
    integer = "int"
    real = "real"


@dataclass
class ParameterBounds:
    min: float | int
    max: float | int
    type: ParameterTypeEnum
    step: int | float | None = None
    log: bool = False


# jsonargparse does not seem to support recursive type defs, so support up to one level of nesting
optimization_space_type = dict[str, list | ParameterBounds]


@dataclass
class Backbone:
    backbone: str
    model_factory: str = "PrithviModelFactory"
    backbone_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    name: str
    type: TaskTypeEnum
    bands: list[HLSBands | int]
    datamodule: BaseDataModule
    decoder: str
    loss: str
    metric: str = "val/loss"
    lr: float = 1e-3
    max_epochs: int = 100
    freeze_backbone: bool = False
    num_classes: int | None = None
    backbone_args: dict[str, Any] = field(default_factory=dict)
    decoder_args: dict[str, Any] = field(default_factory=dict)
    head_args: dict[str, Any] = field(default_factory=dict)
    ignore_index: int | None = None
    optimization_except: set[str] = field(default_factory=set)
