"""Pipeline steps. Signatures are bare on purpose: arguments are not decided yet."""

from dotenv import load_dotenv
from prefect import task, get_run_logger

from pipeline.config import Config
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from matplotlib.figure import Figure
import numpy as np
from pathlib import Path
import pandas as pd
from trainers.base import Trainer
import mlflow
from mlflow import MlflowClient

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
    macro-averaged over classes. Also returns test_roc_auc (macro one-vs-rest)
    and logs one-vs-rest ROC curves as roc_curve.png to the active MLflow run.
    """
    y_true = np.asarray(trainer.y_test)
    y_pred = np.asarray(trainer.predict())
    classes = list(trainer.classes)
    proba = np.asarray(trainer.predict_proba())

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    roc_auc = float(roc_auc_score(
        y_true, proba, multi_class="ovr", average="macro", labels=classes
    ))

    # One-vs-rest ROC curve per class. Figure (not pyplot) needs no GUI backend.
    y_bin = label_binarize(y_true, classes=classes)
    fig = Figure(figsize=(8, 7))
    ax = fig.subplots()
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        class_auc = roc_auc_score(y_bin[:, i], proba[:, i])
        ax.plot(fpr, tpr, lw=1.5, label=f"{cls} (AUC = {class_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"One-vs-rest ROC curves (macro AUC = {roc_auc:.3f})")
    ax.legend(loc="lower right", fontsize="small")
    mlflow.log_figure(fig, "roc_curve.png")

    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_roc_auc": roc_auc,
    }

@task
def register(metrics: dict, trainer: Trainer, registry_name: str) -> str:
    """Log metrics and model, register a new version, and gate @champion on test_f1.

    The new version is registered first (register_model creates `registry_name`
    if it is missing). If there is no @champion (a brand-new registry, or the
    alias was removed), every version, the new one included, is compared on
    its run's logged test_f1 and the best becomes @champion. Otherwise the new
    version (challenger) is promoted only if it beats the champion's test_f1.
    Returns the new version number.
    """
    logger = get_run_logger()
    client = MlflowClient()

    mlflow.log_metrics(metrics)
    model_uri = trainer.log_model()
    new_version = mlflow.register_model(model_uri, registry_name).version

    champion_version = client.get_registered_model(registry_name).aliases.get("champion")
    if champion_version is None:
        versions = client.search_model_versions(f"name = '{registry_name}'")
        f1_by_version = {v.version: client.get_run(v.run_id).data.metrics["test_f1"] for v in versions}
        best = max(f1_by_version, key=f1_by_version.get)
        client.set_registered_model_alias(registry_name, "champion", best)
        if len(versions) == 1:
            logger.info(f"Created registered model {registry_name!r}; version {best} is the first champion.")
        else:
            logger.info(
                f"No champion in {registry_name!r}; compared {len(versions)} versions and crowned "
                f"version {best} (test_f1 {f1_by_version[best]:.4f})."
            )
        return new_version

    champion = client.get_model_version(registry_name, champion_version)
    champion_f1 = client.get_run(champion.run_id).data.metrics["test_f1"]
    challenger_f1 = metrics["test_f1"]

    if challenger_f1 > champion_f1:
        client.set_registered_model_alias(registry_name, "champion", new_version)
        logger.info(
            f"Promoted version {new_version} as the new champion "
            f"(Challenger's test_f1 {challenger_f1:.4f} > Champion test_f1 {champion_f1:.4f} of version {champion.version})."
        )
    else:
        logger.info(
            f"Champion remains version {champion.version} "
            f"(Champion's test_f1 {champion_f1:.4f} >= Challenger's test_f1 {challenger_f1:.4f} of version {new_version})."
        )
    return new_version
