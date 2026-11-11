from trainers.base import Trainer


class YoloTrainer(Trainer):
    """YOLO classification (-cls) variant."""

    def load(self):
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
