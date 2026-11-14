from pipeline.config import Config
from sklearn.linear_model import LogisticRegression
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

    def load(self):
        self.model = LogisticRegression(**self.config.params)

    def preprocess(self, manifest_df):
        self.X_train, self.y_train = load_split(manifest_df, "train")
        self.X_test, self.y_test = load_split(manifest_df, "test")
    
    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        return self.model.predict(self.X_test)
