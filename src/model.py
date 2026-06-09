"""
LightGBM model for cross-sectional stock ranking.

Trained on the most recent 2 years of data, validated on the last 3 months.
Predicts which stocks will outperform their peers over the next 7 days.
"""

import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

LGBM_PARAMS = {
    "objective":       "regression",
    "metric":          "rmse",
    "n_estimators":    1000,
    "learning_rate":   0.02,
    "num_leaves":      63,
    "max_depth":       6,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq":    5,
    "lambda_l1":       0.1,
    "lambda_l2":       1.0,
    "min_child_samples": 50,
    "n_jobs":          -1,
    "verbose":         -1,
    "random_state":    42,
}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("fwd_return", "target")]


def train_final_model(
    feature_matrix: pd.DataFrame,
    train_days: int = 756,   # 3 years
    val_days:   int = 126,   # 6 months
) -> tuple[lgb.LGBMRegressor, list[str]]:
    """Train on most recent data and save model to disk."""
    feature_cols = _feature_cols(feature_matrix)
    dates = feature_matrix.index.get_level_values("date").unique().sort_values()

    train_dates = dates[-(train_days + val_days):-val_days]
    val_dates   = dates[-val_days:]

    train_df = feature_matrix.loc[train_dates]
    val_df   = feature_matrix.loc[val_dates]

    X_train, y_train = train_df[feature_cols].values, train_df["target"].values
    X_val,   y_val   = val_df[feature_cols].values,   val_df["target"].values

    mask_tr = ~np.isnan(y_train) & ~np.any(np.isnan(X_train), axis=1)
    mask_va = ~np.isnan(y_val)   & ~np.any(np.isnan(X_val),   axis=1)

    print(f"  Training: {train_dates[0].date()} → {train_dates[-1].date()} "
          f"({mask_tr.sum()} samples)")
    print(f"  Validating on: {val_dates[0].date()} → {val_dates[-1].date()}")

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_train[mask_tr], y_train[mask_tr],
        eval_set=[(X_val[mask_va], y_val[mask_va])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        feature_name=feature_cols,
    )

    path = MODEL_DIR / "model_latest.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols}, path)
    print(f"  Model saved → {path}  "
          f"(best iteration: {model.best_iteration_})")

    return model, feature_cols


def load_model() -> tuple[lgb.LGBMRegressor, list[str]]:
    path = MODEL_DIR / "model_latest.joblib"
    if not path.exists():
        raise FileNotFoundError(
            "No trained model found. Run: python main.py"
        )
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_cols"]
