from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from loguru import logger
from sklearn.metrics import mean_squared_error

from .config import ExperimentConfig
from .io import save_leaderboard_to_mlflow
from .utils import sanitize_mlflow_name


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
    config: ExperimentConfig
) -> dict:
    """
    使用滑動視窗 (Sliding Window) 評估所有的 AutoGluon TimeSeries 模型，
    並將各個 Window 的指標與平均指標記錄至 MLflow。

    Returns:
        dict: 巢狀字典，包含所有模型在不同 Window 的預測結果。
              格式: { "ModelName": { "window_0": preds_df, "window_1": preds_df, ... } }
    """
    data_cfg = config.data
    ag_cfg = config.autogluon

    pred_len = ag_cfg.prediction_length
    stride = ag_cfg.evaluation_stride
    cov_names = data_cfg.known_covariates_names or []

    # 為了確保切分正確，針對每個 item_id 建立時間序列的索引 (0, 1, 2...)
    test_df = test_df.copy()
    test_df = test_df.sort_values([data_cfg.item_id_col, data_cfg.timestamp_col])
    test_df['__test_idx'] = test_df.groupby(data_cfg.item_id_col).cumcount()
    max_test_size = test_df['__test_idx'].max() + 1

    # 【新增】取得所有訓練好的模型名稱，避免 AutoGluon 只用 best_model 的警告
    models = predictor.model_names()
    logger.info(f"Found {len(models)} models to evaluate: {models}")

    all_model_predictions = {m: {} for m in models}
    model_metrics = {m: [] for m in models}

    # 滑動視窗迴圈
    for w_idx, start_offset in enumerate(range(0, max_test_size - pred_len + 1, stride)):
        window_name = f"window_{w_idx}"
        logger.info(f"Evaluating {window_name} (offset={start_offset})")

        # 1. 準備截至當前的歷史資料 (train_df + test_df 的前半部)
        history_mask = test_df['__test_idx'] < start_offset
        history_add = test_df[history_mask].drop(columns=['__test_idx'])
        current_history_df = pd.concat([train_df, history_add], ignore_index=True)

        ts_history = TimeSeriesDataFrame.from_data_frame(
            current_history_df,
            id_column=data_cfg.item_id_col,
            timestamp_column=data_cfg.timestamp_col
        )

        # 2. 準備預測區間的 Known Covariates 與 Ground Truth (實際值)
        future_mask = (test_df['__test_idx'] >= start_offset) & (test_df['__test_idx'] < start_offset + pred_len)
        current_future_df = test_df[future_mask].drop(columns=['__test_idx'])

        known_covariates = None
        if cov_names:
            known_covariates = TimeSeriesDataFrame.from_data_frame(
                current_future_df[[data_cfg.item_id_col, data_cfg.timestamp_col] + cov_names],
                id_column=data_cfg.item_id_col,
                timestamp_column=data_cfg.timestamp_col
            )

        # 3. 針對每一個模型進行預測與評估
        for model_name in models:
            # 明確指定預測使用的模型，消除警告
            preds = predictor.predict(
                ts_history,
                known_covariates=known_covariates,
                model=model_name
            )
            all_model_predictions[model_name][window_name] = preds

            # --- 計算指標 (RMSE) ---
            preds_df = preds.reset_index()

            # 【關鍵修復】將 AutoGluon 強制的欄位名稱轉回設定檔中的名稱 (例如: trip_date)
            preds_df = preds_df.rename(columns={
                "item_id": data_cfg.item_id_col,
                "timestamp": data_cfg.timestamp_col
            })

            merged = preds_df.merge(
                current_future_df,
                on=[data_cfg.item_id_col, data_cfg.timestamp_col],
                how='inner'
            )

            if not merged.empty:
                # 使用 np.sqrt 計算 RMSE 以兼容不同版本的 sklearn
                rmse = np.sqrt(mean_squared_error(merged[data_cfg.target_col], merged["mean"]))
                model_metrics[model_name].append(rmse)

                # 處理模型名稱中的特殊符號，確保 MLflow 標籤格式合法
                safe_model_name = sanitize_mlflow_name(model_name)
                mlflow.log_metric(f"{safe_model_name}_window_rmse", rmse, step=w_idx)

    # 4. 記錄整體平均指標
    for model_name, metrics in model_metrics.items():
        if metrics:
            avg_rmse = np.mean(metrics)
            safe_model_name = sanitize_mlflow_name(model_name)
            mlflow.log_metric(f"{safe_model_name}_avg_rmse", avg_rmse)
            logger.success(f"Model [{model_name}] Average RMSE: {avg_rmse:.4f}")

    logger.info("Sliding window evaluation completed for all models.")
    return all_model_predictions
