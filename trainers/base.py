from abc import ABC, abstractmethod
from collections.abc import Sequence


class Trainer(ABC):
    """Contract every model must implement to run in the pipeline.

    Future work: an abstract evaluate() so metrics become model-specific.

    The evaluate task also reads `y_test`, set by the trainer: the true labels
    of the test split, in the same order as the predictions.
    """

    y_test: Sequence

    @abstractmethod
    def load(self):
        """Build the model and load its starting weights (pretrained or champion)."""

    @abstractmethod
    def preprocess(self, manifest_df):
        """Prepare the split dataset for this model.

        `manifest_df` has columns path, label, split (train / val / test).
        """

    @abstractmethod
    def train(self):
        """Train the model."""

    @abstractmethod
    def predict(self):
        """Predict class labels for the test split."""
