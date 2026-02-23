import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery  # noqa: F401
from loguru import logger

MAIN_PATH = Path(__file__).resolve().parents[1]
os.environ['MAIN_PATH'] = str(MAIN_PATH)
load_dotenv()
sys.path.append(str(MAIN_PATH))


from src.config import load_config
from src.paths import MODEL_DIR, PLOTS_DIR, ROOT
from src.plot import (
    plot_feature_importance,
    plot_forecast_for_testing,
    plot_leaderboard,
)
from src.preprocess import split_data
from src.trainer import run_autogluon_test, run_autogluon_train


def main():
    # 1. 初始化路徑與配置
    config = load_config(ROOT / "config.yml")
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run() as run:
        current_run_id = run.info.run_id
        current_run_name = run.info.run_name
        logger.info(f"Experiment started. Run ID: {current_run_id}, Run Name: {current_run_name}")

        # 2. 載入資料
        df = pd.read_parquet(ROOT / config.data.data_path)
        df[config.data.item_id_col] = df[config.data.item_id_col].astype(str)  # NOTE: 之後再從sql改

        # 3. 分割資料 (新增 timestamp_col 和 item_id_col 參數)
        train_df, test_df = split_data(df=df, config=config)

        # 4. 執行訓練 (新增 timestamp_col 和 item_id_col 參數)
        predictor, leaderboard = run_autogluon_train(
            train_df=train_df,
            config=config,
            output_path=MODEL_DIR / current_run_name
        )

        predictions = run_autogluon_test(
            predictor=predictor,
            train_df=train_df,
            test_df=test_df,
            config=config
        )

        # 5. 繪製圖表
        plot_forecast_for_testing(train_df, test_df, predictions, config, PLOTS_DIR)
        plot_feature_importance(predictor, test_df, PLOTS_DIR)
        plot_leaderboard(leaderboard, PLOTS_DIR)

        # 6. 記錄 Artifacts
        mlflow.log_artifact(str(ROOT / "config.yml"))
        leaderboard.to_json(ROOT / "data" / "leaderboard.json")
        logger.success("Experiment finished and tracked to MLflow.")


if __name__ == "__main__":
    main()
