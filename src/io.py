import mlflow


def save_leaderboard_to_mlflow(leaderboard):
    best_score = leaderboard.iloc[0]["score_val"]
    best_model = leaderboard.iloc[0]["model"]

    mlflow.log_text(leaderboard.to_markdown(), "autogluon_leaderboard.md")
    mlflow.log_metric("ag_best_score", best_score)
    mlflow.set_tag("ag_best_model", best_model)
