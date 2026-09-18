"""
Runs inside a SageMaker Processing job. Reads the raw labeled dataset,
does basic cleaning, and splits it into train/validation/test so the
same split is used consistently by tuning and evaluation.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_DIR = "/opt/ml/processing/input"
TRAIN_DIR = "/opt/ml/processing/train"
VALIDATION_DIR = "/opt/ml/processing/validation"
TEST_DIR = "/opt/ml/processing/test"


def main() -> None:
    input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    if not input_files:
        raise FileNotFoundError(f"No CSV input found in {INPUT_DIR}")

    df = pd.concat([pd.read_csv(os.path.join(INPUT_DIR, f)) for f in input_files], ignore_index=True)

    df = df.dropna(subset=["label"])
    df = df.fillna(df.median(numeric_only=True))

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    for directory in (TRAIN_DIR, VALIDATION_DIR, TEST_DIR):
        os.makedirs(directory, exist_ok=True)

    train_df.to_csv(os.path.join(TRAIN_DIR, "train.csv"), index=False, header=False)
    val_df.to_csv(os.path.join(VALIDATION_DIR, "validation.csv"), index=False, header=False)
    test_df.to_csv(os.path.join(TEST_DIR, "test.csv"), index=False, header=False)

    print(f"train={len(train_df)} validation={len(val_df)} test={len(test_df)}")


if __name__ == "__main__":
    main()
