import pandas as pd
from sklearn.model_selection import train_test_split


def test_stratified_split_preserves_label_balance():
    df = pd.DataFrame({"label": [0, 1] * 50, "feature_1": range(100)})

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    for split in (train_df, val_df, test_df):
        balance = split["label"].mean()
        assert 0.4 <= balance <= 0.6

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
