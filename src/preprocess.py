import pandas as pd
from loguru import logger

from .config import ExperimentConfig


def split_data(df: pd.DataFrame, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    將時間序列資料依據 item_id 進行分組，並按時間序列切分為訓練集與測試集。

    Args:
        df: 包含原始時間序列資料的 DataFrame。
        config: 全域設定物件 (ExperimentConfig)。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 分割後的 (train_df, test_df)。

    Raises:
        ValueError: 當資料中缺少必要的 item_id、timestamp 或 target 欄位時觸發。
    """
    data_cfg = config.data
    split_cfg = config.split

    logger.info(f"Splitting data with test_size={split_cfg.test_size} per item_id")

    # 1. 檢查必要欄位是否存在
    required_cols = [data_cfg.item_id_col, data_cfg.timestamp_col, data_cfg.target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")

    df = df.copy()

    # 2. 型別轉換與排序
    df[data_cfg.timestamp_col] = pd.to_datetime(df[data_cfg.timestamp_col])
    df = df.sort_values([data_cfg.item_id_col, data_cfg.timestamp_col])

    # 3. 向量化群組切分 (Vectorized Group Split)
    # 計算每個 item_id 群組的總資料量
    group_sizes = df.groupby(data_cfg.item_id_col)[data_cfg.item_id_col].transform('size')
    # 計算每筆資料在該群組中的時間序列索引 (0, 1, 2...)
    row_nums = df.groupby(data_cfg.item_id_col).cumcount()

    # 依據 test_size 計算每個群組的切分門檻，並建立訓練集遮罩
    split_threshold = (group_sizes * (1 - split_cfg.test_size)).astype(int)
    train_mask = row_nums < split_threshold

    # 4. 產出結果
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()

    logger.info(f"Total Train size: {len(train_df)}, Total Test size: {len(test_df)}")

    return train_df, test_df
