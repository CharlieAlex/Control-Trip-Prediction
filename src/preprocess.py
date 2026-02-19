from typing import Optional

import pandas as pd
from loguru import logger


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[str] = None,
    timestamp_col: str = "timestamp",  # 新增
    item_id_col: str = "item_id",  # 新增
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分割時間序列資料"""
    logger.info(f"Splitting data with test_size={test_size}")

    # 確保必要欄位存在
    required_cols = [timestamp_col, target_col]  # noqa: F841

    # 如果沒有 item_id,自動新增
    if item_id_col not in df.columns:
        logger.warning(f"'{item_id_col}' column not found, adding default item_id")
        df = df.copy()
        df[item_id_col] = "item_0"  # AutoGluon 需要 item_id

    # 如果沒有 timestamp,嘗試從 index 取得
    logger.info(df.columns)
    logger.info(timestamp_col)
    if timestamp_col not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            logger.warning(f"'{timestamp_col}' column not found, using index as timestamp")
            df = df.reset_index(names=[timestamp_col])
        else:
            raise ValueError(f"Data must have a '{timestamp_col}' column or DatetimeIndex")

    # 確保 timestamp 是 datetime 型別
    df['timestamp'] = pd.to_datetime(df[timestamp_col])

    # 按時間排序
    df = df.sort_values([item_id_col, 'timestamp'])

    # 對時間序列資料使用時間切分而非隨機切分
    if stratify:
        logger.warning("Time series data should use temporal split, ignoring stratify parameter")

    # 按時間切分
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    return train_df, test_df
