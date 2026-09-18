"""
Runs inside a SageMaker Processing job after tuning. Loads the winning
model artifact and the held-out test split, computes metrics, and writes
evaluation.json in the sagemaker "model quality report" shape so the
pipeline's ConditionStep and the Model Registry can both read it.
"""
import json
import os
import tarfile

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = "/opt/ml/processing/model"
TEST_DIR = "/opt/ml/processing/test"
OUTPUT_DIR = "/opt/ml/processing/evaluation"


def main() -> None:
    tar_path = os.path.join(MODEL_DIR, "model.tar.gz")
    if os.path.exists(tar_path):
        with tarfile.open(tar_path) as tar:
            tar.extractall(MODEL_DIR)

    model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))

    test_files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith(".csv")]
    test_df = pd.concat([pd.read_csv(f, header=None) for f in test_files], ignore_index=True)
    X_test, y_test = test_df.iloc[:, 1:], test_df.iloc[:, 0]

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    report = {
        "binary_classification_metrics": {
            "auc": {"value": roc_auc_score(y_test, y_pred_proba), "standard_deviation": "NaN"},
            "accuracy": {"value": accuracy_score(y_test, y_pred), "standard_deviation": "NaN"},
            "precision": {"value": precision_score(y_test, y_pred), "standard_deviation": "NaN"},
            "recall": {"value": recall_score(y_test, y_pred), "standard_deviation": "NaN"},
        }
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
