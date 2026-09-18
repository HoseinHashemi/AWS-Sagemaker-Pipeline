"""
Kicks off a SageMaker Ground Truth labeling job against unlabeled data
sitting in the raw S3 bucket, using a private workforce. This is the
"annotation" stage of the lifecycle: raw -> manifest of labeled examples
-> consumed by the preprocessing step.

Run manually (not part of the automated pipeline, since labeling needs
human judgment on when a new batch is ready):
    python create_labeling_job.py --input-manifest s3://.../unlabeled.manifest \
        --output-path s3://.../labeled/ --role-arn <arn> --workteam-arn <arn>
"""
import argparse
import time

import boto3


def create_labeling_job(input_manifest: str, output_path: str, role_arn: str, workteam_arn: str) -> str:
    client = boto3.client("sagemaker")
    job_name = f"ml-pipeline-labeling-{int(time.time())}"

    client.create_labeling_job(
        LabelingJobName=job_name,
        LabelAttributeName="label",
        InputConfig={
            "DataSource": {"S3DataSource": {"ManifestS3Uri": input_manifest}},
            "DataAttributes": {"ContentClassifiers": []},
        },
        OutputConfig={"S3OutputPath": output_path},
        RoleArn=role_arn,
        HumanTaskConfig={
            "WorkteamArn": workteam_arn,
            "UiConfig": {
                "UiTemplateS3Uri": f"{output_path}template.liquid.html",
            },
            # AWS publishes region-specific pre-processing/consolidation Lambda
            # ARNs per built-in task type (bounding box, text classification,
            # etc) - look up the correct pair for your region/task type at
            # https://docs.aws.amazon.com/sagemaker/latest/dg/sms-task-types.html
            "PreHumanTaskLambdaArn": "<PRE_HUMAN_TASK_LAMBDA_ARN_FOR_YOUR_REGION_AND_TASK_TYPE>",
            "TaskKeywords": ["classification"],
            "TaskTitle": "Label training examples",
            "TaskDescription": "Classify each example as positive/negative for the target label",
            "NumberOfHumanWorkersPerDataObject": 1,
            "TaskTimeLimitInSeconds": 300,
            "TaskAvailabilityLifetimeInSeconds": 864000,
            "MaxConcurrentTaskCount": 50,
            "AnnotationConsolidationConfig": {
                "AnnotationConsolidationLambdaArn": "<ANNOTATION_CONSOLIDATION_LAMBDA_ARN_FOR_YOUR_REGION_AND_TASK_TYPE>",
            },
        },
    )
    return job_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--workteam-arn", required=True)
    args = parser.parse_args()

    job = create_labeling_job(args.input_manifest, args.output_path, args.role_arn, args.workteam_arn)
    print(f"Started Ground Truth labeling job: {job}")
