from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from loguru import logger
from sklearn.metrics import mean_squared_error

from .config import ExperimentConfig
from .io import save_leaderboard_to_mlflow


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


def run_autogluon_test(
    predictor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
) -> dict:
    """
    使用滑動視窗評估測試集，並將結果記錄至 MLflow。
    回傳每次 Window 的預測結果與評估指標。
    """
    data_cfg = config.data
    ag_cfg = config.autogluon

    pred_len = ag_cfg.prediction_length
    stride = ag_cfg.evaluation_stride
    cov_names = data_cfg.known_covariates_names or []

    # 為 test_df 建立時序索引，方便切分視窗
    test_df = test_df.copy()
    test_df['__test_idx'] = test_df.groupby(data_cfg.item_id_col).cumcount()
    max_test_size = test_df['__test_idx'].max() + 1

    window_predictions = {}
    window_metrics = []

    # 滑動視窗迴圈
    for w_idx, start_offset in enumerate(range(0, max_test_size - pred_len + 1, stride)):
        logger.info(f"Evaluating Window {w_idx}: offset={start_offset}")

        # 1. 準備截至當前的歷史資料 (train_df + test_df 的前半部)
        history_mask = test_df['__test_idx'] < start_offset
        history_add = test_df[history_mask].drop(columns=['__test_idx'])
        current_history_df = pd.concat([train_df, history_add], ignore_index=True)

        ts_history = TimeSeriesDataFrame.from_data_frame(
            current_history_df,
            id_column=data_cfg.item_id_col,
            timestamp_column=data_cfg.timestamp_col
        )

        # 2. 準備預測區間的 Known Covariates 與 Ground Truth
        future_mask = (test_df['__test_idx'] >= start_offset) & (test_df['__test_idx'] < start_offset + pred_len)
        current_future_df = test_df[future_mask].drop(columns=['__test_idx'])

        known_covariates = None
        if cov_names:
            known_covariates = TimeSeriesDataFrame.from_data_frame(
                current_future_df[[data_cfg.item_id_col, data_cfg.timestamp_col] + cov_names],
                id_column=data_cfg.item_id_col,
                timestamp_column=data_cfg.timestamp_col
            )

        # 3. 進行預測
        preds = predictor.predict(ts_history, known_covariates=known_covariates)
        window_predictions[f"window_{w_idx}"] = preds

        # 4. 計算指標 (RMSE)
        # 將預測結果的 'mean' 與實際值對齊比較
        preds_df = preds.reset_index()

        # 【關鍵修復】將 AutoGluon 預設的 index 名稱轉回設定檔中的名稱
        preds_df = preds_df.rename(columns={
            "item_id": data_cfg.item_id_col,
            "timestamp": data_cfg.timestamp_col
        })

        # 現在可以安全地 merge 了
        merged = preds_df.merge(
            current_future_df,
            on=[data_cfg.item_id_col, data_cfg.timestamp_col],
            how='inner'
        )

        if not merged.empty:
            rmse = mean_squared_error(merged[data_cfg.target_col], merged["mean"])
            window_metrics.append(rmse)
            mlflow.log_metric(f"window_{w_idx}_rmse", rmse)
            logger.info(f"Window {w_idx} RMSE: {rmse:.4f}")

    # 記錄整體平均指標
    if window_metrics:
        avg_rmse = np.mean(window_metrics)
        mlflow.log_metric("avg_sliding_window_rmse", avg_rmse)
        logger.success(f"Completed Sliding Window Eval. Average RMSE: {avg_rmse:.4f}")

    return window_predictions
