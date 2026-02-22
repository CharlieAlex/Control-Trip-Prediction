from pathlib import Path

import mlflow
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from loguru import logger


def run_autogluon_train(
    train_data: pd.DataFrame,
    target: str,
    output_path: Path,
    presets="medium_quality",
    time_limit: int = 15,
    prediction_length: int = 7,
    timestamp_col: str = "timestamp",
    item_id_col: str = "item_id",
    known_covariates_names: list[str] = None
):
    """執行 AutoGluon TimeSeries 訓練"""
    logger.info(f"Starting AutoGluon TimeSeries training on target: {target}")

    # 確保資料格式正確
    required_cols = [item_id_col, timestamp_col, target]
    missing_cols = [col for col in required_cols if col not in train_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 轉換為 TimeSeriesDataFrame
    logger.info(train_data.head())
    logger.info(f"Converting to TimeSeriesDataFrame with columns: {train_data.columns.tolist()}")
    train_data = TimeSeriesDataFrame(train_data)

    # 建立預測器
    predictor = TimeSeriesPredictor(
        target=target,
        prediction_length=prediction_length,
        path=str(output_path),
        freq='D',
        known_covariates_names=known_covariates_names,
    )
    predictor.fit(
        train_data,
        presets=presets,
        time_limit=time_limit,
    )

    # 記錄指標到 MLflow
    leaderboard = predictor.leaderboard(silent=False)
    best_score = leaderboard.iloc[0]["score_val"]
    mlflow.log_text(leaderboard.to_markdown(), "autogluon_leaderboard.md")
    mlflow.log_metric("ag_best_score", leaderboard.iloc[0]["score_val"])
    mlflow.set_tag("ag_best_model", leaderboard.iloc[0]["model"])
    mlflow.log_metric("best_val_score", best_score)

    logger.success(f"Training completed. Best validation score: {best_score}")

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
