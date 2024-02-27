"""
This module contains all the logic for fitting models
"""
import copy
import os
from typing import Any

import lightning.pytorch as pl
import mlflow
import optuna
import torch
from lightning import Callback, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from optuna.integration import PyTorchLightningPruningCallback
from ray import train, tune
from ray.air import CheckpointConfig, RunConfig, ScalingConfig
from ray.train import report
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchgeo.datamodules import BaseDataModule
from torchgeo.trainers import BaseTask

from benchmark.types import (
    Backbone,
    ParameterBounds,
    ParameterTypeEnum,
    Task,
    optimization_space_type,
    valid_task_types,
)

os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"  # disable tune loggers


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


###########################################
########### SINGLE NODE - OPTUNA ##########
###########################################
def launch_training(
    trainer: Trainer,
    task: BaseTask,
    datamodule: BaseDataModule,
    run_name: str,
    metric: str,
    storage_uri: str,
    experiment_name: str,
) -> float:
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        # explicitly log batch_size. Since it is not a model param, it will not be logged
        mlflow.log_param("batch_size", datamodule.batch_size)
        mlflow.pytorch.autolog(log_datasets=False)

        # trainer.logger = MLFlowLogger(
        #     experiment_name=experiment_name,
        #     run_id=run.info.run_id,
        #     save_dir=storage_uri,
        #     log_model=True,
        # )
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


def fit_model(
    backbone: Backbone,
    model_args: dict,
    task: Task,
    lightning_task_class: valid_task_types,
    run_name: str,
    storage_uri: str,
    experiment_name: str,
    trial: optuna.Trial | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    freeze_backbone: bool = False,
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
        optimizer_hparams={"weight_decay": 0.05},
        freeze_backbone=freeze_backbone,
        ignore_index=task.ignore_index,
        scheduler=ReduceLROnPlateau,
    )
    callbacks = [
        RichProgressBar(),
        # EarlyStopping(monitor="val/loss", patience=10),  # let user configure this?
    ]

    if trial is not None:
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val/loss"))

    if save_models:
        callbacks.append(ModelCheckpoint(monitor="val/loss"))
    trainer = Trainer(
        callbacks=callbacks,
        max_epochs=task.max_epochs,
        enable_checkpointing=save_models,
    )
    return launch_training(
        trainer,
        lightning_task,
        task.datamodule,
        run_name,
        task.metric,
        storage_uri,
        experiment_name,
    ), task.metric


def fit_model_with_hparams(
    backbone: Backbone,
    task: Task,
    lightning_task_class: valid_task_types,
    base_args: dict[str, Any],
    run_name: str,
    hparam_space: optimization_space_type,
    storage_uri: str,
    experiment_name: str,
    save_models: bool,
    trial: optuna.Trial,
) -> float:
    current_hparams: dict[str, int | float | str | bool] = {}

    for parameter, space in hparam_space.items():
        if parameter in task.optimization_except:
            continue
        if isinstance(space, list):
            suggestion = trial.suggest_categorical(parameter, space)
            if suggestion is None:
                raise Exception(f"Optuna suggested None for parameter {parameter}")
            current_hparams[parameter] = suggestion
        elif isinstance(space, ParameterBounds):
            match space.type:
                case ParameterTypeEnum.integer:
                    current_hparams[parameter] = trial.suggest_int(
                        parameter,
                        int(space.min),
                        int(space.max),
                    )
                case ParameterTypeEnum.real:
                    current_hparams[parameter] = trial.suggest_float(
                        parameter, space.min, space.max, log=space.log
                    )
                case _:
                    raise Exception(
                        f"Type {space.type} not recognized. Suggest one of {[e.value for e in ParameterTypeEnum]}"
                    )
    lr = float(current_hparams.pop("lr", task.lr))
    batch_size = current_hparams.pop("batch_size", None)
    if batch_size is not None:
        batch_size = int(batch_size)
    freeze_backbone = bool(current_hparams.pop("freeze_backbone", False))
    model_args = inject_hparams(base_args, current_hparams)
    run_name = f"{run_name}_{trial.number}"
    return fit_model(
        backbone,
        model_args,
        task,
        lightning_task_class,
        run_name,
        storage_uri,
        experiment_name,
        trial,
        lr=lr,
        batch_size=batch_size,
        freeze_backbone=freeze_backbone,
        save_models=save_models,
    )[0]  # return only the metric value for optuna


###########################################
########### MULTI NODE - RAY ##############
###########################################


class RayReportCallback(pl.callbacks.Callback):
    """Like Ray's Report Callback but with no checkpointing"""

    def __init__(self) -> None:
        super().__init__()
        self.trial_name = train.get_context().get_trial_name()
        self.local_rank = train.get_context().get_local_rank()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Creates a checkpoint dir with fixed name
        metrics = trainer.callback_metrics
        metrics = {k: v.item() for k, v in metrics.items()}

        # (Optional) Add customized metrics
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step

        # Add a barrier to ensure all workers finished reporting here
        torch.distributed.barrier()
        report(metrics=metrics)


def ray_tune_model(
    backbone: Backbone,
    task: Task,
    lightning_task_class: valid_task_types,
    base_args: dict[str, Any],
    run_name: str,
    hparam_space: optimization_space_type,
    storage_uri: str,
    experiment_name: str,
    save_models: bool,
    num_trials: int,
) -> tune.ResultGrid:
    trainable = tune.with_parameters(
        ray_fit_model,
        backbone=backbone,
        base_args=base_args,
        task=task,
        lightning_task_class=lightning_task_class,
        run_name=run_name,
        storage_uri=storage_uri,
        experiment_name=experiment_name,
        parent_run_id=mlflow.active_run().info.run_id,
        save_models=save_models,
    )

    current_hparams: dict[str, Any] = {}

    for parameter, space in hparam_space.items():
        if parameter in task.optimization_except:
            continue
        if isinstance(space, list):
            suggestion = tune.choice(space)
            if suggestion is None:
                raise Exception(f"Optuna suggested None for parameter {parameter}")
            current_hparams[parameter] = suggestion
        elif isinstance(space, ParameterBounds):
            match space.type:
                case ParameterTypeEnum.integer:
                    current_hparams[parameter] = tune.quniform(space.min, space.max, 1)
                case ParameterTypeEnum.real:
                    if space.log:
                        current_hparams[parameter] = tune.loguniform(
                            space.min, space.max
                        )
                    else:
                        current_hparams[parameter] = tune.uniform(space.min, space.max)
                case _:
                    raise Exception(
                        f"Type {space.type} not recognized. Suggest one of {[e.value for e in ParameterTypeEnum]}"
                    )

    # Early stopping
    scheduler = ASHAScheduler(
        max_t=task.max_epochs, grace_period=min(task.max_epochs, 5), reduction_factor=2
    )

    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 6, "GPU": 1}
    )
    ray_trainer = TorchTrainer(
        trainable,
        scaling_config=scaling_config,
        run_config=RunConfig(
            name=run_name,
            storage_path=os.path.join(storage_uri, "../ray"),
            checkpoint_config=CheckpointConfig(num_to_keep=1, checkpoint_frequency=0),
        ),
    )

    tuner = tune.Tuner(
        ray_trainer,
        tune_config=tune.TuneConfig(
            metric=task.metric,
            mode="min",  # let user choose this
            num_samples=num_trials,
            search_alg=ConcurrencyLimiter(OptunaSearch(), max_concurrent=6),
            scheduler=scheduler,
        ),
        run_config=RunConfig(
            name=run_name,
            storage_path=os.path.join(storage_uri, "../ray"),
            checkpoint_config=CheckpointConfig(num_to_keep=1, checkpoint_frequency=0),
        ),
        param_space={"train_loop_config": current_hparams},
    )

    results = tuner.fit()
    return results


def ray_fit_model(
    config: dict,
    backbone: Backbone,
    base_args: dict,
    task: Task,
    lightning_task_class: valid_task_types,
    run_name: str,
    storage_uri: str,
    experiment_name: str,
    parent_run_id: str,
    save_models: bool = True,
) -> None:
    lr = float(config.pop("lr", task.lr))
    batch_size = config.pop("batch_size", None)
    if batch_size is not None:
        batch_size = int(batch_size)
    freeze_backbone = bool(config.pop("freeze_backbone", False))
    model_args = inject_hparams(base_args, config)
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
        optimizer_hparams={"weight_decay": 0.05},
        freeze_backbone=freeze_backbone,
        ignore_index=task.ignore_index,
        scheduler=ReduceLROnPlateau,
        # scheduler_hparams={"patience": 5},
    )
    callbacks: list[Callback] = [RayReportCallback()]

    if save_models:
        callbacks.append(ModelCheckpoint(monitor="val/loss"))

    trainer = Trainer(
        strategy=RayDDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        plugins=[RayLightningEnvironment()],
        enable_checkpointing=save_models,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=False,
        max_epochs=task.max_epochs,
    )
    trainer = prepare_trainer(trainer)

    mlflow.set_tracking_uri(storage_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(nested=True):
        # hack for nestedness
        mlflow.set_tag("mlflow.parentRunId", parent_run_id)

        mlflow.pytorch.autolog(log_datasets=False, log_models=False)
        # explicitly log batch_size. Since it is not a model param, it will not be logged
        mlflow.log_param("batch_size", task.datamodule.batch_size)
        trainer.fit(lightning_task, datamodule=task.datamodule)
