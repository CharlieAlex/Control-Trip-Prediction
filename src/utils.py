import re


def sanitize_mlflow_name(name: str) -> str:
    """
    將字串轉換為 MLflow 合法的命名格式。
    將所有非字母、數字、下劃線、橫線、點、空格、冒號、斜線的字元替換為底線。
    """
    # 匹配所有不符合 MLflow 規範的字元
    # 規範：alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), colon(:) and slashes (/).
    return re.sub(r'[^a-zA-Z0-9._\- :\/]', '_', name)
