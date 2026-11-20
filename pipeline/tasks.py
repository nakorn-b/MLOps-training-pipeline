"""Pipeline steps. Signatures are bare on purpose: arguments are not decided yet."""

from dotenv import load_dotenv
from prefect import task, get_run_logger

from pipeline.config import Config
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
import pandas as pd
from trainers.base import Trainer
import mlflow

load_dotenv()

@task
def load_config(config_path: str) -> Config:
    """Read the experiment YAML and validate it into a Config."""
    logger = get_run_logger()
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) 

    logger.info(data['data']['uri'])
    logger.info(data['model']['name'])
    return Config(**data)

@task
def preprocess(data_uri: str, trainer: Trainer) -> pd.DataFrame:
    """Split the dataset, then let the trainer prepare it for its model.

    `data_uri` is a folder with one subfolder per class. Images are assigned
    to train / val / test (80/10/10, stratified, fixed seed) in a manifest
    with columns path, label, split; no files are moved. The manifest then
    goes to trainer.preprocess() for the model-specific step.
    """
    seed = 42
    paths = sorted(Path(data_uri).glob("*/*.png"))
    if not paths:
        raise FileNotFoundError(f"no images matching */*.png under {data_uri}")
    labels = [p.parent.name for p in paths]
    train_path, test_path, train_label, test_label = train_test_split(
        paths, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    test_path, val_path, test_label, val_label = train_test_split(
        test_path, test_label, test_size=0.5, random_state=seed, stratify=test_label
    )
    manifest_df = pd.DataFrame({
        "path": [*train_path, *val_path, *test_path],
        "label": [*train_label, *val_label, *test_label],
        "split": ["train"]*len(train_path) + ["val"]*len(val_path) + ["test"]*len(test_path),
        })
    trainer.preprocess(manifest_df)
    return manifest_df

@task
def train(trainer: Trainer):
    """Call trainer.load() then trainer.train()."""
    trainer.load()
    trainer.train()

@task
def evaluate(trainer: Trainer):
    """Score the trainer on its test split.

    Generic pipeline code until Trainer gets its own evaluate().
    Returns accuracy, precision, recall and f1; the last three are
    macro-averaged over classes.
    """
    y_true = np.asarray(trainer.y_test)
    y_pred = np.asarray(trainer.predict())

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

@task
def register(metrics: dict, trainer: Trainer) -> str:
    """
    Log the final metrics and the model to the active run; return the model URI.
    """
    mlflow.log_metrics(metrics)
    return trainer.log_model()
