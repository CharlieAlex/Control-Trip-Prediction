from pathlib import Path
from typing import Optional

from heisenberg_utils.config_utils import load_yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    data_path: str
    target_col: str
    timestamp_col: str = "timestamp"
    item_id_col: str = "item_id"
    known_covariates_names: Optional[list[str]] = None


class SplitConfig(BaseModel):
    test_size: float = 0.2
    random_state: int = 42


class AutoGluonConfig(BaseModel):
    presets: str = "medium_quality"
    time_limit: int = 15
    prediction_length: int = 7
    freq: str = "D"
    eval_metric: str = "WQL"
    random_state: int = 123
    evaluation_stride: int = 5
    plot_history_days: int = 60
    plot_ymin: Optional[float] = None
    plot_ymax: Optional[float] = None


class MLflowConfig(BaseModel):
    tracking_uri: str


class ExperimentConfig(BaseModel):
    experiment_name: str
    data: DataConfig
    split: SplitConfig
    autogluon: AutoGluonConfig
    mlflow: MLflowConfig


def load_config(config_path: Path) -> ExperimentConfig:
    config = load_yaml(config_path)
    return ExperimentConfig(**config)
