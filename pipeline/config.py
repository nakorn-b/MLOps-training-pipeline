from typing import Any

from pydantic import BaseModel


class DataConfig(BaseModel):
    uri: str  # dataset folder with one subfolder per class; the pipeline splits it


class ModelConfig(BaseModel):
    name: str  # key in model_factory.TRAINERS


class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    # Hyperparameters for the chosen model. Keys are model-specific, so the
    # trainer validates them; every key is logged to MLflow as `params.<key>`.
    params: dict[str, Any] = {}
