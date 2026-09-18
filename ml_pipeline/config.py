"""Central place for names/paths shared between the pipeline definition and
infra stacks, so they don't drift out of sync."""

PROJECT_NAME = "ml-sagemaker-pipeline"
MODEL_PACKAGE_GROUP_NAME = f"{PROJECT_NAME}-model-group"
PIPELINE_NAME = f"{PROJECT_NAME}-training-pipeline"
FEATURE_GROUP_NAME = f"{PROJECT_NAME}-features"

# Quality gate: a candidate model must beat this AUC on the held-out test
# set before it is registered as "Approved"-eligible. This is what makes
# the pipeline a *gate*, not just an automation script.
MIN_MODEL_AUC = 0.75
