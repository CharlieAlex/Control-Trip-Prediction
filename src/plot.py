from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from .config import ExperimentConfig
from .io import save_plot_to_mlflow


def plot_forecast(
    predictor,
    test_data,
    predictions,
    plots_dir: Path,
    quantile_levels: list[float] | None = None,
    filename: str = "forecast_plot.png",
    mlflow_subdir: str = "plots",
) -> Path:
    """
    使用 AutoGluon TimeSeriesPredictor.plot() 繪製預測結果圖，
    並儲存到磁碟及 MLflow Artifacts。

    Parameters
    ----------
    predictor : TimeSeriesPredictor
        已訓練完成的 predictor。
    test_data : TimeSeriesDataFrame
        測試集（已轉換為 TimeSeriesDataFrame）。
    predictions : TimeSeriesDataFrame
        由 ``predictor.predict(test_data)`` 產生的預測結果。
    plots_dir : Path
        圖片存放目錄。
    quantile_levels : list[float], optional
        要顯示的分位數區間，預設 ``[0.1, 0.9]``。
    filename : str
        輸出檔名。
    mlflow_subdir : str
        MLflow artifact 子目錄。

    Returns
    -------
    Path
        儲存的圖片路徑。
    """
    if quantile_levels is None:
        quantile_levels = [0.1, 0.9]

    predictor.plot(
        test_data,
        predictions=predictions,
        quantile_levels=quantile_levels,
    )
    fig = plt.gcf()  # predictor.plot() 使用現有的 figure

    out_path = plots_dir / filename
    save_plot_to_mlflow(fig, out_path, mlflow_subdir)
    return out_path


def plot_feature_importance(
    predictor,
    test_data,
    plots_dir: Path,
    top_n: int | None = None,
    filename: str = "feature_importance.png",
    mlflow_subdir: str = "plots",
) -> Path | None:
    """
    呼叫 ``predictor.feature_importance()`` 繪製水平長條圖，
    並儲存到磁碟及 MLflow Artifacts。

    部分模型不支援 feature importance；此情況會記錄 warning 並回傳 ``None``。

    Parameters
    ----------
    predictor : TimeSeriesPredictor
        已訓練完成的 predictor。
    test_data : TimeSeriesDataFrame
        測試集（用於計算重要性）。
    plots_dir : Path
        圖片存放目錄。
    top_n : int, optional
        僅顯示前 N 個重要特徵；``None`` 表示全部顯示。
    filename : str
        輸出檔名。
    mlflow_subdir : str
        MLflow artifact 子目錄。

    Returns
    -------
    Path or None
        儲存的圖片路徑；若不支援則回傳 ``None``。
    """
    try:
        importance: pd.DataFrame = predictor.feature_importance(test_data)
    except Exception as exc:
        logger.warning("Could not generate feature importance: %s", exc)
        return None

    # importance DataFrame 通常包含 "importance" 欄位，以 feature 為 index
    if "importance" not in importance.columns:
        logger.warning(
            "feature_importance() returned unexpected columns: %s", importance.columns.tolist()
        )
        return None

    df_plot = importance.sort_values("importance", ascending=True)
    if top_n is not None:
        df_plot = df_plot.tail(top_n)  # tail = 最重要的 top_n 個（ascending=True 排序下）

    fig, ax = plt.subplots(figsize=(10, max(4, len(df_plot) * 0.4)))
    ax.barh(df_plot.index, df_plot["importance"], color="#4C72B0", edgecolor="white")
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_path = plots_dir / filename
    save_plot_to_mlflow(fig, out_path, mlflow_subdir)
    return out_path


def plot_leaderboard(
    leaderboard: pd.DataFrame,
    plots_dir: Path,
    score_col: str = "score_val",
    model_col: str = "model",
    filename: str = "leaderboard_comparison.png",
    mlflow_subdir: str = "plots",
) -> Path:
    """
    根據 ``leaderboard`` DataFrame 繪製各模型驗證分數比較圖，
    並儲存到磁碟及 MLflow Artifacts。

    此函數取代原本未定義的 ``plot_cross_validation()``，
    功能上等同於「各模型 CV 分數的視覺化比較」。

    Parameters
    ----------
    leaderboard : pd.DataFrame
        由 ``predictor.leaderboard(test_data)`` 產生，
        需包含 ``model_col`` 與 ``score_col`` 欄位。
    plots_dir : Path
        圖片存放目錄。
    score_col : str
        分數欄位名稱，預設 ``"score_val"``。
    model_col : str
        模型名稱欄位，預設 ``"model"``。
    filename : str
        輸出檔名。
    mlflow_subdir : str
        MLflow artifact 子目錄。

    Returns
    -------
    Path
        儲存的圖片路徑。

    Raises
    ------
    KeyError
        若 ``leaderboard`` 缺少必要欄位。
    """
    for col in (model_col, score_col):
        if col not in leaderboard.columns:
            raise KeyError(
                f"leaderboard 缺少欄位 '{col}'，現有欄位：{leaderboard.columns.tolist()}"
            )

    lb = leaderboard[[model_col, score_col]].sort_values(score_col, ascending=True)

    # 依分數高低上色（最高分 = 深色）
    colors = plt.cm.Blues_r(
        [i / max(len(lb) - 1, 1) for i in range(len(lb))]
    )

    fig, ax = plt.subplots(figsize=(10, max(4, len(lb) * 0.5)))
    bars = ax.barh(lb[model_col], lb[score_col], color=colors, edgecolor="white")

    # 在每個 bar 右側標示數值
    for bar, val in zip(bars, lb[score_col]):
        ax.text(
            bar.get_width() + abs(lb[score_col].max()) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    ax.set_title("Model Leaderboard — Validation Scores", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Score ({score_col})")
    ax.set_ylabel("Model")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_path = plots_dir / filename
    save_plot_to_mlflow(fig, out_path, mlflow_subdir)
    return out_path
