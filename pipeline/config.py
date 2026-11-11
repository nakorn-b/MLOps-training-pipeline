from pydantic import BaseModel


class DataConfig(BaseModel):
    uri: str  # local path or remote URI; data is already split into train/ val/ test/


class ModelConfig(BaseModel):
    name: str  # key in model_factory.TRAINERS


class TrainConfig(BaseModel):
    """Training hyperparameters. Fields not decided yet."""


class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    train: TrainConfig = TrainConfig()
