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


def save_plot_to_mlflow(
    fig: plt.Figure,
    local_path: Path,
    mlflow_path: str,
    file_name: str,
) -> None:
    """儲存圖片到本地資料夾，並上傳到 MLflow Artifacts。
    """
    full_path = local_path / file_name  # 組合完整本地路徑
    full_path.parent.mkdir(parents=True, exist_ok=True)

    if full_path.exists():
        full_path.unlink()  # 移除舊檔案

    fig.savefig(full_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    mlflow.log_artifact(str(full_path), artifact_path=mlflow_path)  # log 到指定子路徑
    logger.info(f"Saved and logged to MLflow: {file_name} in {mlflow_path}")
