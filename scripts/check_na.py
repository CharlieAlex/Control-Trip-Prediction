from pathlib import Path

import pandas as pd
from google.cloud import bigquery  # noqa: F401
from loguru import logger

MAIN_PATH = Path(__file__).parents[1]


def check_missing_dates(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    freq: str = 'D'
) -> pd.DatetimeIndex:
    """檢查 DataFrame 中缺漏的日期"""
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    mind = df[timestamp_col].min()
    maxd = df[timestamp_col].max()

    full_range = pd.date_range(start=mind, end=maxd, freq=freq)
    missing_dates = full_range.difference(df[timestamp_col])

    logger.info(f"時間範圍: {mind.date()} ~ {maxd.date()}，共 {len(full_range)} 天")
    logger.info(f"資料筆數: {df.shape[0]} 行，唯一日期數: {df[timestamp_col].nunique()}")
    logger.info(f"缺漏天數: {len(missing_dates)}")

    if len(missing_dates) > 0:
        logger.warning(f"缺漏日期:\n{missing_dates.strftime('%Y-%m-%d').tolist()}")
    else:
        logger.success("無缺漏日期")

    return missing_dates


if __name__ == '__main__':
    df = pd.read_parquet(MAIN_PATH / 'data' / 'data.parquet')
    check_missing_dates(df)
