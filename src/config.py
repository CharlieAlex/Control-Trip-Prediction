from pathlib import Path

from heisenberg_utils.config_utils import load_yaml
from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    experiment_name: str
    data: dict
    split: dict
    autogluon: dict
    mlflow: dict


def load_config(config_path: Path) -> ExperimentConfig:
    config = load_yaml(config_path)
    return ExperimentConfig(**config)
