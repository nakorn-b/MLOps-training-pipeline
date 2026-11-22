from abc import ABC, abstractmethod
from collections.abc import Sequence


class Trainer(ABC):
    """Contract every model must implement to run in the pipeline.

    Future work: an abstract evaluate() so metrics become model-specific.

    The evaluate task also reads `y_test`, set by the trainer: the true labels
    of the test split, in the same order as the predictions. It also reads
    `classes`: the class labels in the column order of predict_proba().
    """

    y_test: Sequence
    classes: Sequence

    @abstractmethod
    def load(self, model_uri: str | None = None):
        """Build the model and load its starting weights.

        `model_uri` None: fresh or pretrained weights. Otherwise an MLflow model
        URI (e.g. `models:/<name>@champion`, `runs:/<run_id>/model`) to start from.
        The caller resolves the URI; trainers never look up the registry.
        """

    @abstractmethod
    def preprocess(self, manifest_df):
        """Prepare the split dataset for this model.

        `manifest_df` has columns path, label, split (train / val / test).
        """

    @abstractmethod
    def train(self):
        """Train the model.

        Runs inside the flow's active MLflow run, so it may log metrics
        (e.g. per epoch) with mlflow.log_metric.
        """

    @abstractmethod
    def predict(self):
        """Predict class labels for the test split."""

    @abstractmethod
    def predict_proba(self):
        """Predict class probabilities for the test split.

        Shape (n_test, n_classes), columns in `self.classes` order.
        """

    @abstractmethod
    def log_model(self) -> str:
        """Log the trained model to the active MLflow run, in this model's flavor.

        Returns the logged model's URI. Called by the register task inside an
        open run, so the pipeline never needs to know the flavor.
        """
