from model_factory import get_trainer
from prefect import flow

from pipeline import tasks
import pandas as pd
import mlflow
import os
from datetime import datetime

from trainers.base import Trainer

@flow
def train(config_path: str):
    """Train from pretrained weights on the dataset named in the YAML.

    Gating: compare against the champion's logged metric.
    Everything after load_config logs into one MLflow run, opened here.

    Flow: preprocess -> train -> evaluate -> register
    """
    cfg = tasks.load_config(config_path)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(os.environ.get("EXPERIMENT_NAME", "MLOps-pipeline"))
    run_name = f"{cfg.model.name}-{datetime.now():%Y-%m-%d-%H:%M:%S}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(pd.json_normalize(cfg.model_dump()).iloc[0].to_dict())
        trainer = get_trainer(cfg)
        tasks.preprocess(cfg.data.uri, trainer)
        tasks.train(trainer)
        metrics = tasks.evaluate(trainer)
        tasks.register(metrics, trainer, cfg.model.registry_name)


@flow
def retrain(config_path: str, model_uri: str | None = None, dataset_uri: str | None = None):
    """Fine-tune an existing model, optionally on a new dataset.

    The model to start from is `model_uri` if given, else `model.uri` from the
    YAML; any MLflow model URI works (a registry version, @champion, a run's
    model). Fails before any work if neither is set. `dataset_uri`, if given,
    overrides `data.uri` the same way.
    Gating: compare against the champion's logged metric (same as train for
    now; scoring both on the new test split is planned).

    Flow: preprocess -> train (init from model_uri) -> evaluate -> register
    """
    cfg = tasks.load_config(config_path)
    # CLI/flow parameters override the YAML, so the logged params show what ran.
    if model_uri:
        cfg.model.uri = model_uri
    if dataset_uri:
        cfg.data.uri = dataset_uri
    if not cfg.model.uri:
        raise ValueError(
            "retrain needs a model to fine-tune: set model.uri in "
            f"{config_path} or pass --model-uri"
        )

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(os.environ.get("EXPERIMENT_NAME", "MLOps-pipeline"))
    run_name = f"retrain-{cfg.model.name}-{datetime.now():%Y-%m-%d-%H:%M:%S}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(pd.json_normalize(cfg.model_dump()).iloc[0].to_dict())
        trainer: Trainer = get_trainer(cfg) # TODO: change this to cfg.model.name
        tasks.preprocess(cfg.data.uri, trainer)
        tasks.train(trainer, cfg.model.uri)
        metrics = tasks.evaluate(trainer)
        tasks.register(metrics, trainer, cfg.model.registry_name)