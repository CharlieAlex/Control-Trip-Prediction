from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
from autogluon.timeseries import TimeSeriesDataFrame

matplotlib.use("Agg")  # 非互動式後端，避免 GUI 相依
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 共用工具
# ---------------------------------------------------------------------------

def _save_and_log(fig: plt.Figure, path: Path, mlflow_subdir: str = "plots") -> None:
    """儲存圖片到磁碟，並上傳到 MLflow Artifacts。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path=mlflow_subdir)
    logger.info("Saved and logged to MLflow: %s", path.name)


def prepare_timeseries_data(df: pd.DataFrame, date_col: str = "trip_date") -> TimeSeriesDataFrame:
    """
    將一般 DataFrame 轉換為 AutoGluon TimeSeriesDataFrame。

    Parameters
    ----------
    df : pd.DataFrame
        原始資料，需包含 ``date_col`` 欄位。
    date_col : str
        日期欄位名稱，預設 ``"trip_date"``。

    Returns
    -------
    TimeSeriesDataFrame
    """

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data[date_col])
    return TimeSeriesDataFrame(data)


# ---------------------------------------------------------------------------
# 圖 1：Forecast Plot
# ---------------------------------------------------------------------------

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
    _save_and_log(fig, out_path, mlflow_subdir)
    return out_path


def plot_forecast_with_actual(
    predictor,
    data: pd.DataFrame,
    predictions,                   # ← 外部傳入，不在函數內重新 predict
    plots_dir: Path,
    quantile_levels: list[float] | None = None,
    target_col: str = "trip_per_user",
    item_ids: list | None = None,
    n_context_days: int = 30,
    horizon: int = 10,
    filename: str = "forecast_with_actual.png",
    mlflow_subdir: str = "plots",
) -> Path:
    if quantile_levels is None:
        quantile_levels = [0.1, 0.9]

    ts_data = TimeSeriesDataFrame.from_data_frame(
        data,
        id_column="item_id",
        timestamp_column="timestamp",
    )

    all_ids = ts_data.item_ids.tolist()
    ids_to_plot = item_ids if item_ids is not None else all_ids[:6]

    n = len(ids_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), squeeze=False)

    for ax, item_id in zip(axes[:, 0], ids_to_plot):
        mask = ts_data.index.get_level_values(0) == item_id
        full_df = ts_data[mask]
        full_ts = full_df.index.get_level_values(1)
        full_vals = full_df[target_col].to_numpy()

        # forecast 的時間點
        mask_pred = predictions.index.get_level_values(0) == item_id
        pred_df = predictions[mask_pred]
        pred_ts = pred_df.index.get_level_values(1)
        pred_mean = pred_df["mean"] if "mean" in pred_df.columns else pred_df.iloc[:, 0]

        # actual：從 full_data 找出和 pred_ts 時間點完全一致的列
        full_ts_index = pd.DatetimeIndex(full_ts)
        actual_mask = full_ts_index.isin(pred_ts)
        actual_ts = full_ts[actual_mask]
        actual_vals = full_vals[actual_mask]

        # 歷史：pred 開始之前的 n_context_days 天
        history_mask = full_ts_index < pred_ts[0]
        history_ts = full_ts[history_mask][-n_context_days:]
        history_vals = full_vals[history_mask][-n_context_days:]

        ax.plot(history_ts, history_vals, label="History",
                color="gray", linewidth=1.2, alpha=0.6)
        ax.plot(actual_ts, actual_vals, label="Actual",
                color="steelblue", linewidth=2.0)
        ax.plot(pred_ts, pred_mean.values, label="Forecast",
                color="tomato", linewidth=1.8, linestyle="--")

        if len(quantile_levels) >= 2:
            q_low = str(quantile_levels[0])
            q_high = str(quantile_levels[-1])
            if q_low in pred_df.columns and q_high in pred_df.columns:
                ax.fill_between(
                    pred_ts,
                    pred_df[q_low].values,
                    pred_df[q_high].values,
                    alpha=0.2, color="tomato",
                    label=f"PI [{q_low}–{q_high}]",
                )

        if len(actual_ts) > 0:
            ax.axvline(x=actual_ts[0], color="black", linestyle=":", linewidth=1.0, alpha=0.7)

        ax.set_title(f"Item: {item_id}")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"Forecast vs Actual (last {horizon} days)",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = plots_dir / filename
    _save_and_log(fig, out_path, mlflow_subdir)
    return out_path

# ---------------------------------------------------------------------------
# 圖 2：Feature Importance
# ---------------------------------------------------------------------------


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
    _save_and_log(fig, out_path, mlflow_subdir)
    return out_path


# ---------------------------------------------------------------------------
# 圖 3：Leaderboard Comparison（取代 plot_cross_validation）
# ---------------------------------------------------------------------------

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
    _save_and_log(fig, out_path, mlflow_subdir)
    return out_path
