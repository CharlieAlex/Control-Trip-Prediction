from pathlib import Path

import mlflow
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from loguru import logger
from .config import ExperimentConfig


def run_autogluon_train(
    train_df: pd.DataFrame,
    config: ExperimentConfig,
    output_path: Path
):
    """執行 AutoGluon TimeSeries 訓練"""
    # 建立捷徑變數，方便後續呼叫
    config_data = config.data
    config_ag = config.autogluon

    # 確保資料格式正確
    required_cols = {config_data.item_id_col, config_data.timestamp_col, config_data.target_col}
    missing_cols = required_cols - set(train_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    # 轉換資料
    ts_dataframe = TimeSeriesDataFrame.from_data_frame(
        train_df,
        id_column=config_data.item_id_col,
        timestamp_column=config_data.timestamp_col
    )

    # 建立模型
    predictor = TimeSeriesPredictor(
        target=config_data.target_col,
        prediction_length=config_ag.prediction_length,
        path=str(output_path),
        freq=config_ag.freq,
        known_covariates_names=config_data.known_covariates_names,
    )

    # 開始訓練
    predictor.fit(
        ts_dataframe,
        presets=config_ag.presets,
        time_limit=config_ag.time_limit,
    )

    # 記錄指標到 MLflow
    leaderboard = predictor.leaderboard(silent=False)

    if leaderboard.empty:
        raise ValueError("Leaderboard is empty. No models were successfully trained.")

    save_leaderboard_to_mlflow(leaderboard)

    return predictor, leaderboard


def run_autogluon_test(predictor, test_df, config):

    cov_names = config.data.get("known_covariates_names", [])
    time_col = "timestamp" if "timestamp" in test_df.columns else "trip_date"

    test_data = TimeSeriesDataFrame.from_data_frame(
        test_df,
        id_column="item_id",
        timestamp_column=time_col
    )

    known_covariates = None
    if cov_names:
        # 用 make_future_data_frame 生成正確的未來時間點 index
        future_df = predictor.make_future_data_frame(test_data)

        # 將 test_df 的 covariate 值 merge 進去
        cov_source = test_df[["item_id", time_col] + cov_names].copy()
        future_df = future_df.reset_index()
        future_df = future_df.merge(
            cov_source,
            on=["item_id", time_col],
            how="left"
        )

        known_covariates = TimeSeriesDataFrame.from_data_frame(
            future_df,
            id_column="item_id",
            timestamp_column=time_col
        )

    predictions = predictor.predict(test_data, known_covariates=known_covariates)

    return predictions
