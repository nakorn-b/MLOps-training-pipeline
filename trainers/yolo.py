from trainers.base import Trainer


class YoloTrainer(Trainer):
    """YOLO classification (-cls) variant."""

    def load(self, model_uri: str | None = None):
        raise NotImplementedError

    def preprocess(self, manifest_df):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def log_model(self) -> str:
        raise NotImplementedError
