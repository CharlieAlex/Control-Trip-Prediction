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
from src.paths import MODEL_DIR, ROOT
from src.plot import (
    plot_feature_importance,
    plot_forecast,
    plot_forecast_with_actual,
    plot_leaderboard,
)
from src.preprocess import split_data
from src.trainer import run_autogluon_test, run_autogluon_train


def main():
    # 1. 初始化路徑與配置
    config = load_config(ROOT / "config.yml")
    mlflow.set_tracking_uri(config.mlflow["tracking_uri"])
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run() as run:
        current_run_id = run.info.run_id
        current_run_name = run.info.run_name
        logger.info(f"Experiment started. Run ID: {current_run_id}, Run Name: {current_run_name}")

        # 2. 載入資料
        df = pd.read_parquet(ROOT / config.data["data_path"])

        # 3. 分割資料 (新增 timestamp_col 和 item_id_col 參數)
        train_df, test_df = split_data(
            df=df,
            target_col=config.data["target_col"],
            test_size=config.split["test_size"],
            random_state=config.split["random_state"],
            stratify=config.split.get("stratify"),
            timestamp_col=config.data.get("timestamp_col", "timestamp"),
            item_id_col=config.data.get("item_id_col", "item_id"),
        )

        # 4. 執行訓練 (新增 timestamp_col 和 item_id_col 參數)
        _, leaderboard = run_autogluon_train(
            train_data=train_df,
            target=config.data["target_col"],
            output_path=MODEL_DIR / current_run_name,
            presets=config.autogluon["presets"],
            prediction_length=config.autogluon["prediction_length"],
            timestamp_col=config.data.get("timestamp_col", "timestamp"),
            item_id_col=config.data.get("item_id_col", "item_id"),
        )

        # 5. 繪製圖表
        plots_dir = ROOT / "plots"
        predictions = run_autogluon_test(
            predictor=predictor,
            test_df=test_df,
            config=config
        )
        plot_forecast(predictor, test_df, predictions, plots_dir)
        plot_forecast_with_actual(predictor, df, predictions, plots_dir)
        plot_feature_importance(predictor, test_df, plots_dir)
        plot_leaderboard(leaderboard, plots_dir)

        # 6. 記錄 Artifacts
        mlflow.log_artifact(str(ROOT / "config.yml"))
        leaderboard.to_json(ROOT / "data" / "leaderboard.json")
        logger.success("Experiment finished and tracked to MLflow.")


if __name__ == "__main__":
    main()
