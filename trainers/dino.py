from trainers.base import Trainer


class DinoTrainer(Trainer):
    """DINOv3 backbone with a classifier head."""

    def load(self, model_uri: str | None = None):
        raise NotImplementedError

    def preprocess(self, manifest_df):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def predict_proba(self):
        raise NotImplementedError

    def log_model(self) -> str:
        raise NotImplementedError
