import mlflow
from pipeline.config import Config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from trainers.base import Trainer
import pandas as pd
import numpy as np
from PIL import Image

def load_split(df: pd.DataFrame, split: str):
    rows = df[df['split'] == split]
    X = np.stack([
        np.asarray(Image.open(p).convert("RGB").resize((32, 32))).flatten() / 255.0
        for p in rows["path"]
    ])
    y = rows['label'].to_numpy()
    return X, y

class LogisticRegressionTrainer(Trainer):
    def __init__(self, config: Config) -> None:
        self.config = config

    def load(self, model_uri: str | None = None):
        if model_uri is None:
            self.model = LogisticRegression(**self.config.params)
        else:
            self.model = mlflow.sklearn.load_model(model_uri)

    def preprocess(self, manifest_df):
        self.X_train, self.y_train = load_split(manifest_df, "train")
        self.X_test, self.y_test = load_split(manifest_df, "test")
    
    def train(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_train)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_train, y_pred, average="macro", zero_division=0
        )
        mlflow.log_metrics({
            "train_accuracy": float(accuracy_score(self.y_train, y_pred)),
            "train_precision": float(precision),
            "train_recall": float(recall),
            "train_f1": float(f1),
        })

    @property
    def classes(self):
        return self.model.classes_

    def predict(self):
        return self.model.predict(self.X_test)

    def predict_proba(self):
        return self.model.predict_proba(self.X_test) # sklearn method

    def log_model(self) -> str:
        info = mlflow.sklearn.log_model(
            self.model, 
            name=self.config.model.name, 
            input_example=self.X_test[:1],
        )
        return info.model_uri
