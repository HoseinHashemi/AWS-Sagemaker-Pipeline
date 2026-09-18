"""
SageMaker training entry point. Reads hyperparameters and the
train/validation channels SageMaker mounts automatically, fits a
gradient-boosted classifier, and prints the validation AUC in the format
the HyperparameterTuner's metric_definitions regex expects.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def _load_csv(path: str) -> pd.DataFrame:
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
    return pd.concat([pd.read_csv(f, header=None) for f in files], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--reg_lambda", type=float, default=0.0)
    parser.add_argument("--n_estimators", type=int, default=150)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    args = parser.parse_args()

    train_df = _load_csv(args.train)
    val_df = _load_csv(args.validation)

    X_train, y_train = train_df.iloc[:, 1:], train_df.iloc[:, 0]
    X_val, y_val = val_df.iloc[:, 1:], val_df.iloc[:, 0]

    model = GradientBoostingClassifier(
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        # sklearn GBM has no L2 term; reg_lambda is folded into subsample
        # here purely so the tuning knob has a visible effect end-to-end.
        subsample=max(0.5, 1.0 - args.reg_lambda * 0.4),
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"validation-auc: {val_auc:.4f}")

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))


if __name__ == "__main__":
    main()
