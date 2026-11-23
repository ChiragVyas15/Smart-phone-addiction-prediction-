"""Train a CatBoost classifier and create cat_sub.csv for this competition."""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
TARGET = "addicted_label"
ID_COLUMN = "id"


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add consistent, missing-value-safe behavioural features to a data frame."""
    out = frame.copy()
    screen = out["daily_screen_time_hours"]
    social = out["social_media_hours"]
    gaming = out["gaming_hours"]
    sleep = out["sleep_hours"]
    opens = out["app_opens_per_day"]

    out["total_screen_activity"] = screen + social + gaming
    out["weekend_extra_screen"] = out["weekend_screen_time"] - screen
    out["screen_sleep_ratio"] = screen / (sleep + 1.0)
    out["social_screen_ratio"] = social / (screen + 1.0)
    out["gaming_screen_ratio"] = gaming / (screen + 1.0)
    out["notification_app_ratio"] = out["notifications_per_day"] / (opens + 1.0)
    return out


def prepare_features(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    out = add_features(frame).drop(columns=[ID_COLUMN], errors="ignore")
    if TARGET in out:
        out = out.drop(columns=[TARGET])
    if feature_columns is not None:
        out = out.reindex(columns=feature_columns)

    # CatBoost accepts numeric NaNs natively; categorical values must be strings.
    for column in out.select_dtypes(include=["object", "string", "category"]).columns:
        out[column] = out[column].fillna("__MISSING__").astype(str)
    return out


def main() -> None:
    root = Path(__file__).resolve().parent
    train = pd.read_csv(root / "train.csv")
    test = pd.read_csv(root / "test.csv")
    print(f"Loaded train={train.shape}, test={test.shape}", flush=True)

    X = prepare_features(train)
    y = train[TARGET].astype(int)
    cat_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    print(f"Prepared {X.shape[1]} features ({len(cat_features)} categorical)", flush=True)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=RANDOM_SEED
    )

    print("Fitting validation model...", flush=True)
    validation_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=1200,
        learning_rate=0.08,
        depth=8,
        l2_leaf_reg=6.0,
        random_strength=0.5,
        random_seed=RANDOM_SEED,
        thread_count=6,
        verbose=100,
        allow_writing_files=False,
    )
    validation_model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100,
        verbose=100,
    )

    valid_probabilities = validation_model.predict_proba(X_valid)[:, 1]
    baseline_probability = float(y_train.mean())
    print(f"Validation log loss: {log_loss(y_valid, valid_probabilities):.6f}")
    print(f"Constant-baseline log loss: {log_loss(y_valid, np.full(len(y_valid), baseline_probability)):.6f}")
    print(f"Validation ROC-AUC: {roc_auc_score(y_valid, valid_probabilities):.6f}")

    best_iterations = validation_model.get_best_iteration() + 1
    print(f"Refitting on all labelled rows with {best_iterations} iterations.", flush=True)
    final_model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=best_iterations,
        learning_rate=0.08,
        depth=8,
        l2_leaf_reg=6.0,
        random_strength=0.5,
        random_seed=RANDOM_SEED,
        thread_count=6,
        verbose=100,
        allow_writing_files=False,
    )
    final_model.fit(X, y, cat_features=cat_features)

    X_test = prepare_features(test, feature_columns=X.columns.tolist())
    predictions = final_model.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({ID_COLUMN: test[ID_COLUMN], TARGET: predictions})
    output_path = root / "cat_sub.csv"
    submission.to_csv(output_path, index=False)

    assert len(submission) == len(test)
    assert submission[TARGET].between(0, 1).all()
    print(f"Wrote {output_path.name}: {submission.shape[0]} rows")
    print(submission[TARGET].describe().to_string())


if __name__ == "__main__":
    main()
