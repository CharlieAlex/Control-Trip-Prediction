import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from .config import ExperimentConfig
from .io import save_plot_to_mlflow
from .utils import sanitize_mlflow_name


def plot_forecast(
    predictor,
    test_data,
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

    predictions = predictor.predict(test_data)
    predictor.plot(
        test_data,
        predictions=predictions,
        quantile_levels=quantile_levels,
    )
    fig = plt.gcf()  # predictor.plot() 使用現有的 figure

    save_plot_to_mlflow(fig, plots_dir, mlflow_subdir, filename)

    return plots_dir / filename


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

    save_plot_to_mlflow(fig, plots_dir, mlflow_subdir, filename)

    return plots_dir / filename


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

    save_plot_to_mlflow(fig, plots_dir, mlflow_subdir, filename)

    return plots_dir / filename


def plot_forecast_for_testing(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_model_predictions: dict,
    config: ExperimentConfig,
    plots_dir: Path,
    mlflow_subdir: str = "plots",
) -> list[Path]:
    """
    繪製時間序列預測圖表 (Grid 版本 + 局部時間放大)。
    只顯示預測開始前 N 天與預測期間的實際值與預測值。
    """
    data_cfg = config.data
    ag_cfg = config.autogluon
    sns.set_theme(style="whitegrid")

    full_actuals = pd.concat([train_df, test_df], ignore_index=True)
    items = full_actuals[data_cfg.item_id_col].unique()

    saved_paths = []

    for model_name, window_predictions in all_model_predictions.items():
        for item in items:
            item_actuals = full_actuals[full_actuals[data_cfg.item_id_col] == item]

            n_windows = len(window_predictions)
            ncols = min(2, n_windows)
            nrows = math.ceil(n_windows / ncols) if n_windows > 0 else 1

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(15 * ncols, 5 * nrows),
                squeeze=False
            )
            axes = axes.flatten()

            for w_idx, (window_name, preds) in enumerate(window_predictions.items()):
                ax = axes[w_idx]

                if item in preds.index.get_level_values(0):
                    item_preds = preds.loc[item].reset_index()
                    item_preds = item_preds.rename(columns={"timestamp": data_cfg.timestamp_col})

                    # 【優化】計算該 window 的時間範圍
                    pred_start = item_preds[data_cfg.timestamp_col].min()
                    pred_end = item_preds[data_cfg.timestamp_col].max()
                    history_start = pred_start - pd.Timedelta(days=ag_cfg.plot_history_days)

                    # 【優化】篩選該 window 專屬的實際值範圍 (歷史 60 天 + 預測 N 天)
                    window_actuals = item_actuals[
                        (item_actuals[data_cfg.timestamp_col] >= history_start) &
                        (item_actuals[data_cfg.timestamp_col] <= pred_end)
                    ]

                    # 1. 畫截取後的實際值 (黑色實線)
                    sns.lineplot(
                        data=window_actuals,
                        x=data_cfg.timestamp_col,
                        y=data_cfg.target_col,
                        ax=ax,
                        label="Actual Target",
                        color="black",
                        linewidth=1.5,
                        linestyle="-"
                    )

                    # 2. 畫預測值 (紅色虛線)
                    sns.lineplot(
                        data=item_preds,
                        x=data_cfg.timestamp_col,
                        y="mean",
                        ax=ax,
                        label="Prediction",
                        color="red",
                        linestyle="--",
                        linewidth=2
                    )

                    # 3. 畫 90% 信賴區間
                    if "0.1" in item_preds.columns and "0.9" in item_preds.columns:
                        ax.fill_between(
                            item_preds[data_cfg.timestamp_col],
                            item_preds["0.1"],
                            item_preds["0.9"],
                            color="red",
                            alpha=0.2
                        )

                    # 【優化】設定 X 軸顯示範圍，確保畫面比例固定
                    ax.set_xlim(history_start, pred_end)
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
                    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

                ax.set_title(f"{window_name}", fontsize=12)
                ax.set_xlabel("Time")
                ax.set_ylabel(data_cfg.target_col)
                ax.legend(loc='upper left')

            # 隱藏空白子圖
            for empty_idx in range(n_windows, len(axes)):
                fig.delaxes(axes[empty_idx])

            plt.suptitle(f"Model: {model_name} | Item: {item}", fontsize=16, y=1.02)
            plt.tight_layout()

            # 儲存
            safe_model_name = sanitize_mlflow_name(model_name)
            filename = f"forecast_{safe_model_name}_{item}.png"

            save_plot_to_mlflow(fig, plots_dir, mlflow_subdir, filename)
            saved_paths.append(plots_dir / filename)

            plt.close(fig)

    return saved_paths
