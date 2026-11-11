from trainers import DinoTrainer, Trainer, YoloTrainer

# Maps the YAML `model.name` to a Trainer class. Classes, not instances:
# only the requested trainer is ever constructed.
TRAINERS: dict[str, type[Trainer]] = {
    "dinov3": DinoTrainer,
    "yolo": YoloTrainer,
}


def get_trainer():
    """Instantiate the Trainer registered under the configured model name.

    An unknown name must raise an error listing the valid names in TRAINERS.
    """
    raise NotImplementedError
