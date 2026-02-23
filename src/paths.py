from pathlib import Path


def get_project_root() -> Path:
    """自動偵測專案根目錄 (以 pyproject.toml 為基準)"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent  # Fallback


# 常數定義
ROOT = get_project_root()
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
EXPERIMENT_DIR = ROOT / "experiments"
QUERIES_DIR = ROOT / "src" / "queries"
SA_DIR = ROOT / "sa"
PLOTS_DIR = ROOT / "plots"
