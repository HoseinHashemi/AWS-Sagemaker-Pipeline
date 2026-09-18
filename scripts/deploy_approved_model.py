"""
Finds the most recently Approved model package in the registry and
deploys it to a real-time endpoint. Reusing the same endpoint name across
deployments (instead of creating a new one each time) lets SageMaker do a
built-in blue/green update: it stands up the new variant, health-checks
it, shifts traffic, and only then tears down the old one - so a bad
deploy never causes a visible outage.
"""
import argparse
import time

import boto3


def get_latest_approved_package(client, model_package_group_name: str) -> dict:
    response = client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    packages = response["ModelPackageSummaryList"]
    if not packages:
        raise RuntimeError(f"No Approved model packages found in {model_package_group_name}")
    return packages[0]


def deploy(model_package_group_name: str, endpoint_name: str, role_arn: str) -> None:
    client = boto3.client("sagemaker")
    package_summary = get_latest_approved_package(client, model_package_group_name)
    package_arn = package_summary["ModelPackageArn"]
    model_name = f"ml-pipeline-model-{int(time.time())}"

    client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        Containers=[{"ModelPackageName": package_arn}],
    )

    endpoint_config_name = f"{endpoint_name}-config-{int(time.time())}"
    client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": "ml.m5.large",
                "InitialInstanceCount": 1,
            }
        ],
        DataCaptureConfig={
            # Captures live inference requests/responses so Model Monitor
            # has real traffic to compare against the training baseline.
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": f"s3://{endpoint_name}-data-capture/",
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
        },
    )

    existing = client.list_endpoints(NameContains=endpoint_name)["Endpoints"]
    if any(e["EndpointName"] == endpoint_name for e in existing):
        print(f"Updating existing endpoint {endpoint_name} (SageMaker performs a blue/green rollout)")
        client.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
    else:
        print(f"Creating new endpoint {endpoint_name}")
        client.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)

    print(f"Deployed model package {package_arn} to endpoint {endpoint_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-package-group-name", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--role-arn", required=True)
    args = parser.parse_args()
    deploy(args.model_package_group_name, args.endpoint_name, args.role_arn)
