from trainers.base import Trainer


class DinoTrainer(Trainer):
    """DINOv3 backbone with a classifier head."""

    def load(self):
        raise NotImplementedError

    def preprocess(self, manifest_df):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
