from abc import ABC, abstractmethod


class Trainer(ABC):
    """Contract every model must implement to run in the pipeline.

    Future work: an abstract evaluate() so metrics become model-specific.
    """

    @abstractmethod
    def load(self):
        """Build the model and load its starting weights (pretrained or champion)."""

    @abstractmethod
    def preprocess(self):
        """Convert the pre-split dataset into the format this model expects."""

    @abstractmethod
    def train(self):
        """Train the model."""

    @abstractmethod
    def predict(self):
        """Predict class labels for images."""
