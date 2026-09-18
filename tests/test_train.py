import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def test_model_beats_random_on_separable_data():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=50, random_state=42)
    model.fit(X, y)

    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    assert auc > 0.9


def test_inference_output_shape_matches_input_rows():
    from src.inference.inference import predict_fn

    class DummyModel:
        def predict_proba(self, X):
            return np.column_stack([1 - X[:, 0], X[:, 0]])

    X = np.array([[0.2], [0.8], [0.5]])
    preds = predict_fn(X, DummyModel())
    assert preds.shape == (3,)
