from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from loguru import logger


def save_leaderboard_to_mlflow(leaderboard):
    best_score = leaderboard.iloc[0]["score_val"]
    best_model = leaderboard.iloc[0]["model"]

    mlflow.log_text(leaderboard.to_markdown(), "autogluon_leaderboard.md")
    mlflow.log_metric("ag_best_score", best_score)
    mlflow.set_tag("ag_best_model", best_model)


def save_plot_to_mlflow(fig: plt.Figure, path: Path, mlflow_subdir: str = "plots") -> None:
    """儲存圖片到磁碟，並上傳到 MLflow Artifacts。"""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path=mlflow_subdir)
    logger.info("Saved and logged to MLflow: %s", path.name)
