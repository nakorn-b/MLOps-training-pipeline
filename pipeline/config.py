from typing import Any

from pydantic import BaseModel


class DataConfig(BaseModel):
    uri: str  # dataset folder with one subfolder per class; the pipeline splits it


class ModelConfig(BaseModel):
    name: str  # key in model_factory.TRAINERS
    registry_name: str  # MLflow registered model that holds this experiment's versions
    # Retrain only: MLflow model URI to fine-tune from, e.g. models:/<registry_name>/3
    # or models:/<registry_name>@champion. The --model-uri CLI flag overrides it.
    uri: str | None = None


class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    # Hyperparameters for the chosen model. Keys are model-specific, so the
    # trainer validates them; every key is logged to MLflow as `params.<key>`.
    params: dict[str, Any] = {}
