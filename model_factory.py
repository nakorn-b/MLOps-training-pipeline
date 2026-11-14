from pipeline.config import Config
from trainers import DinoTrainer, Trainer, YoloTrainer, LogisticRegressionTrainer

# Maps the YAML `model.name` to a Trainer class. Classes, not instances:
# only the requested trainer is ever constructed.
TRAINERS: dict[str, type[Trainer]] = {
    "dinov3": DinoTrainer,
    "yolo": YoloTrainer,
    "logistic_regression": LogisticRegressionTrainer
}


def get_trainer(model_name: str, config: Config) -> Trainer:
    """Instantiate the Trainer registered under the configured model name.

    An unknown name must raise an error listing the valid names in TRAINERS.
    """
    if model_name not in TRAINERS:
        raise ValueError(f"unknown model {model_name!r}; valid names: {sorted(TRAINERS)}")
    return TRAINERS[model_name](config)
