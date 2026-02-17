import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

MAIN_PATH = Path(__file__).resolve().parents[1]
os.environ['MAIN_PATH'] = str(MAIN_PATH)
load_dotenv()
sys.path.append(str(MAIN_PATH))

from heisenberg_utils.bigquery_utils import path_to_df

from src.paths import DATA_DIR, QUERIES_DIR


def get_data() -> pd.DataFrame:
    df = path_to_df(
        script_path=QUERIES_DIR / 'data.sql',
        save_path=DATA_DIR / 'data.parquet',
        is_confirm=False,
    )

    return df


if __name__ == "__main__":
    df = get_data()
