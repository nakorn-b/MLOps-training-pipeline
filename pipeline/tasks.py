"""Pipeline steps. Signatures are bare on purpose: arguments are not decided yet."""

from prefect import task

from pipeline.config import Config
import yaml


@task
def load_config(config_path: str) -> Config:
    """Read the experiment YAML and validate it into a Config."""
    with open(config_path, "r") as f:
        data = yaml.save_load(f) 

    print(data['data']['uri'])
    print(data['model']['name'])

@task
def preprocess():
    """Delegate to trainer.preprocess(). Never splits data."""
    raise NotImplementedError


@task
def train():
    """Call trainer.load() then trainer.train()."""
    raise NotImplementedError


@task
def evaluate():
    """Run trainer.predict() on the test split and compute accuracy / F1.

    Generic pipeline code until Trainer gets its own evaluate().
    """
    raise NotImplementedError


@task
def register():
    """Log the run to MLflow; promote to champion only if it beats the current one."""
    raise NotImplementedError
