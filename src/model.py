"""
Walk-forward LightGBM model for cross-sectional stock ranking.

Walk-forward prevents lookahead bias: the model is always trained on
past data only and evaluated on future data it has never seen.

Architecture:
  - Train window: 504 trading days (~2 years)
  - Validation window: 63 trading days (~3 months, used for early stopping)
  - Test window: 21 trading days (~1 month, held out)
  - Slide forward 21 days, retrain, repeat
"""

import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 1000,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": 6,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "min_child_samples": 50,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("fwd_return", "target")]


def train_single_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> lgb.LGBMRegressor:
    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["target"].values

    mask_train = ~np.isnan(y_train) & ~np.any(np.isnan(X_train), axis=1)
    mask_val = ~np.isnan(y_val) & ~np.any(np.isnan(X_val), axis=1)

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_train[mask_train],
        y_train[mask_train],
        eval_set=[(X_val[mask_val], y_val[mask_val])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        feature_name=feature_cols,
    )
    return model


def walk_forward_train(
    feature_matrix: pd.DataFrame,
    train_days: int = 504,
    val_days: int = 63,
    test_days: int = 21,
) -> tuple[pd.DataFrame, list[lgb.LGBMRegressor], pd.DataFrame]:
    """
    Run full walk-forward cross-validation.

    Returns:
        predictions: DataFrame with columns [date, ticker, pred, actual, fwd_return]
        models: list of trained models (one per fold)
        feature_importance: mean importance across all folds
    """
    feature_cols = _get_feature_cols(feature_matrix)
    dates = feature_matrix.index.get_level_values("date").unique().sort_values()

    start_idx = train_days + val_days
    all_preds = []
    models = []
    importances = []

    total_folds = (len(dates) - start_idx) // test_days
    print(f"Running {total_folds} walk-forward folds ({test_days}d test windows)...")

    fold = 0
    for test_start_idx in range(start_idx, len(dates) - test_days + 1, test_days):
        train_dates = dates[test_start_idx - train_days - val_days : test_start_idx - val_days]
        val_dates = dates[test_start_idx - val_days : test_start_idx]
        test_dates = dates[test_start_idx : test_start_idx + test_days]

        train_df = feature_matrix.loc[train_dates]
        val_df = feature_matrix.loc[val_dates]
        test_df = feature_matrix.loc[test_dates]

        # Skip if not enough data
        if len(train_df) < 500 or len(val_df) < 50:
            continue

        model = train_single_fold(train_df, val_df, feature_cols)
        models.append(model)

        # Predict on test set
        X_test = test_df[feature_cols].values
        preds = model.predict(X_test)

        result = test_df[["fwd_return", "target"]].copy()
        result["pred"] = preds
        all_preds.append(result)

        # Feature importance
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        importances.append(imp)

        fold += 1
        if fold % 5 == 0:
            print(f"  Fold {fold}/{total_folds} — test period: {test_dates[0].date()} → {test_dates[-1].date()}")

    predictions = pd.concat(all_preds)
    feature_importance = pd.DataFrame(importances).mean().sort_values(ascending=False)

    print(f"\nWalk-forward complete. {len(predictions)} predictions across {fold} folds.")
    return predictions, models, feature_importance


def train_final_model(
    feature_matrix: pd.DataFrame,
    train_days: int = 504,
    val_days: int = 63,
) -> lgb.LGBMRegressor:
    """Train a final model on the most recent data for live signal generation."""
    feature_cols = _get_feature_cols(feature_matrix)
    dates = feature_matrix.index.get_level_values("date").unique().sort_values()

    train_dates = dates[-(train_days + val_days) : -val_days]
    val_dates = dates[-val_days:]

    train_df = feature_matrix.loc[train_dates]
    val_df = feature_matrix.loc[val_dates]

    print(f"Training final model: {train_dates[0].date()} → {val_dates[-1].date()}")
    model = train_single_fold(train_df, val_df, feature_cols)

    path = MODEL_DIR / "model_latest.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols}, path)
    print(f"Saved to {path}")
    return model, feature_cols


def load_model() -> tuple[lgb.LGBMRegressor, list[str]]:
    path = MODEL_DIR / "model_latest.joblib"
    if not path.exists():
        raise FileNotFoundError("No trained model found. Run train.py first.")
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_cols"]


def evaluate_predictions(predictions: pd.DataFrame) -> dict:
    """Compute IC (information coefficient) and top-decile hit rate."""
    results = {}

    # Information Coefficient: Spearman correlation between predicted rank and actual rank
    ic_series = (
        predictions.groupby(level="date")
        .apply(
            lambda g: g["pred"].corr(g["target"], method="spearman")
            if len(g) > 5
            else np.nan
        )
        .dropna()
    )
    results["mean_ic"] = ic_series.mean()
    results["ic_ir"] = ic_series.mean() / ic_series.std()  # IC information ratio
    results["ic_positive_pct"] = (ic_series > 0).mean()

    # Top-decile hit rate: how often does top-decile predicted end up in top-decile actual?
    def top_decile_hit(g):
        if len(g) < 10:
            return np.nan
        top_pred = g.nlargest(max(1, len(g) // 10), "pred").index
        top_actual = g.nlargest(max(1, len(g) // 10), "fwd_return").index
        return len(set(top_pred) & set(top_actual)) / len(top_pred)

    hit_rate = (
        predictions.groupby(level="date")
        .apply(top_decile_hit)
        .dropna()
        .mean()
    )
    results["top_decile_hit_rate"] = hit_rate

    return results
