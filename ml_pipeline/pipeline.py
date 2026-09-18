"""
The SageMaker Pipeline definition: this is the backbone of the whole
project. One Python function builds a DAG of steps that SageMaker
Pipelines executes and tracks (lineage, caching, retries) for us.

DAG:
  Preprocess (Processing)
      -> Tune (Hyperparameter Tuning, picks the best of N training jobs)
          -> Evaluate (Processing: compute AUC/precision/recall on test set)
              -> ClarifyCheck (bias + feature-attribution baseline)
              -> ConditionStep: AUC >= MIN_MODEL_AUC?
                    yes -> RegisterModel (into Model Registry, PendingManualApproval)
                    no  -> pipeline fails here, nothing bad gets registered

Run locally with:  python ml_pipeline/pipeline.py --role-arn <execution-role-arn>
"""
import argparse

import setuptools  # noqa: F401  (must import before sagemaker: shims distutils on Python 3.12+)
import sagemaker
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.clarify import BiasConfig, DataConfig
from sagemaker.workflow.clarify_check_step import ClarifyCheckStep, DataBiasCheckConfig
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TuningStep
from sagemaker.workflow.step_collections import RegisterModel

from config import MIN_MODEL_AUC, MODEL_PACKAGE_GROUP_NAME, PIPELINE_NAME


def build_pipeline(role_arn: str, artifacts_bucket: str, region: str) -> Pipeline:
    session = PipelineSession(default_bucket=artifacts_bucket)

    # --- Parameters: values you'd override per pipeline execution without
    # editing code (e.g. new raw data location, instance size for a bigger
    # dataset, or a tighter quality bar before a big release). ---
    input_data_uri = ParameterString(
        name="InputDataUri", default_value=f"s3://{artifacts_bucket}/raw/dataset.csv"
    )
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge")
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
    min_auc_threshold = ParameterFloat(name="MinModelAUC", default_value=MIN_MODEL_AUC)
    max_tuning_jobs = ParameterInteger(name="MaxTuningJobs", default_value=6)

    # --- Step 1: Preprocess. Splits raw data into train/validation/test,
    # handles missing values / encoding, writes back to S3 as columnar
    # data the training container can stream. ---
    sklearn_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="preprocess",
        role=role_arn,
        sagemaker_session=session,
    )
    step_preprocess = ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        code="src/processing/preprocess.py",
        inputs=[ProcessingInput(source=input_data_uri, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
    )

    # --- Step 2: Hyperparameter tuning. Instead of training once, we launch
    # a small search over learning rate / regularization and let SageMaker's
    # Bayesian tuner pick the winner. This is Automatic Model Tuning. ---
    estimator = SKLearn(
        entry_point="src/training/train.py",
        framework_version="1.2-1",
        instance_type=training_instance_type,
        instance_count=1,
        role=role_arn,
        base_job_name="train",
        sagemaker_session=session,
        use_spot_instances=True,     # cheaper training; safe here since jobs are short + idempotent
        max_wait=3600,
        max_run=1800,
    )
    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="validation:auc",
        objective_type="Maximize",
        hyperparameter_ranges={
            "learning_rate": ContinuousParameter(0.01, 0.3),
            "reg_lambda": ContinuousParameter(0.0, 1.0),
        },
        max_jobs=max_tuning_jobs,
        max_parallel_jobs=2,
        metric_definitions=[{"Name": "validation:auc", "Regex": r"validation-auc: ([0-9\.]+)"}],
    )
    step_tune = TuningStep(
        name="TuneModel",
        tuner=tuner,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
            ),
            "validation": sagemaker.inputs.TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri
            ),
        },
    )
    best_model_artifact = step_tune.get_top_model_s3_uri(top_k=0, s3_bucket=artifacts_bucket)

    # --- Step 3: Evaluate the winning model against the untouched test
    # split, and write a metrics.json that both the ConditionStep and the
    # Model Registry entry (for lineage / model cards) read from. ---
    eval_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="evaluate",
        role=role_arn,
        sagemaker_session=session,
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        code="src/processing/evaluate.py",
        inputs=[
            ProcessingInput(source=best_model_artifact, destination="/opt/ml/processing/model"),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        property_files=[evaluation_report],
    )

    # --- Step 4: Clarify bias/explainability baseline. This is what a
    # Model Monitor "bias drift" schedule compares future traffic against
    # after the model is deployed. Only runs once the model has already
    # cleared the quality gate (it's nested inside the ConditionStep below),
    # so we don't spend a Clarify job on a model that's going to be rejected.
    clarify_check_config = DataBiasCheckConfig(
        data_config=DataConfig(
            s3_data_input_path=step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            s3_output_path=f"s3://{artifacts_bucket}/clarify-output",
            label="label",
            headers=["label", "feature_1", "feature_2", "feature_3"],
            dataset_type="text/csv",
        ),
        # Checks whether the positive label is imbalanced across a
        # sensitive attribute (feature_1 here) - facet_values_or_threshold
        # left unset means "compare each value of the facet against the rest".
        data_bias_config=BiasConfig(label_values_or_threshold=[1], facet_name="feature_1"),
    )
    check_job_config = CheckJobConfig(
        role=role_arn,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        sagemaker_session=session,
    )
    step_clarify_check = ClarifyCheckStep(
        name="ClarifyBiasBaseline",
        clarify_check_config=clarify_check_config,
        check_job_config=check_job_config,
        skip_check=True,       # first run: nothing to compare against yet, just record the baseline
        register_new_baseline=True,
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"{step_evaluate.properties.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri}/evaluation.json",
            content_type="application/json",
        )
    )

    # --- Step 5: Register — but only behind a quality gate. ---
    step_register = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=best_model_artifact,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    step_condition = ConditionStep(
        name="CheckModelQuality",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=step_evaluate.name,
                    property_file=evaluation_report,
                    json_path="binary_classification_metrics.auc.value",
                ),
                right=min_auc_threshold,
            )
        ],
        if_steps=[step_clarify_check, step_register],
        else_steps=[],  # no else step -> pipeline execution ends without registering
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            input_data_uri,
            processing_instance_type,
            training_instance_type,
            min_auc_threshold,
            max_tuning_jobs,
        ],
        steps=[step_preprocess, step_tune, step_evaluate, step_condition],
        sagemaker_session=session,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--artifacts-bucket", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--upsert", action="store_true", help="create/update the pipeline in SageMaker")
    parser.add_argument("--start", action="store_true", help="start a new pipeline execution")
    args = parser.parse_args()

    pipeline = build_pipeline(args.role_arn, args.artifacts_bucket, args.region)
    print(pipeline.definition())

    if args.upsert:
        pipeline.upsert(role_arn=args.role_arn)
    if args.start:
        execution = pipeline.start()
        print(f"Started execution: {execution.arn}")
