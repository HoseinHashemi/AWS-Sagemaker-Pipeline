"""
Custom inference handlers for the SKLearn model server used by the
real-time endpoint and batch transform jobs. SageMaker's SKLearn
container calls these four functions.
"""
import json
import os

import joblib
import numpy as np


def model_fn(model_dir: str):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        rows = [list(map(float, line.split(","))) for line in request_body.strip().split("\n")]
        return np.array(rows)
    if request_content_type == "application/json":
        payload = json.loads(request_body)
        return np.array(payload["instances"])
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    return model.predict_proba(input_data)[:, 1]


def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps({"predictions": prediction.tolist()}), accept
    return ",".join(map(str, prediction.tolist())), "text/csv"
