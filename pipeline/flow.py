from model_factory import get_trainer
from prefect import flow

from pipeline import tasks
import pandas as pd
import mlflow
import os

@flow
def train(config_path: str):
    """Train from pretrained weights on the dataset named in the YAML.

    Gating: compare against the champion's logged metric.
    Everything after load_config logs into one MLflow run, opened here.
    """
    cfg = tasks.load_config(config_path)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(os.environ.get("EXPERIMENT_NAME", "MLOps-pipeline"))
    with mlflow.start_run(run_name=cfg.model.name):
        mlflow.log_params(pd.json_normalize(cfg.model_dump()).iloc[0].to_dict())
        trainer = get_trainer(cfg)
        tasks.preprocess(cfg.data.uri, trainer)
        tasks.train(trainer)
        metrics = tasks.evaluate(trainer)
        tasks.register(metrics, trainer)   
    # TODO: wire once task arguments are decided
    # preprocess -> train -> evaluate -> register


@flow
def retrain(config_path: str, dataset_uri: str | None = None):
    """Fine-tune the current champion, optionally on a new dataset.

    Fails if no champion exists (run `train` first).
    Gating: score champion and new model on the same (new) test split.
    """
    cfg = tasks.load_config(config_path)
    if dataset_uri:
        cfg.data.uri = dataset_uri
    # TODO: wire once task arguments are decided
    # preprocess -> train (init from champion) -> evaluate (both models) -> register
    raise NotImplementedError