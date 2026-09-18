#!/usr/bin/env python3
"""
CDK app entry point. Wires the whole architecture together into a single
deployable stack (one CloudFormation stack composed of constructs, so
everything is created/destroyed atomically for this learning project;
split into multiple cdk.Stacks if you need independent lifecycles later).

Deploy with:
    cdk deploy --context github_owner=<you> --context github_repo=<repo> \
        --context github_connection_arn=<arn> --context notification_email=<email>
"""
import os

import aws_cdk as cdk

from infra.cicd_stack import CicdStack
from infra.data_stack import DataStack
from infra.deploy_stack import DeployStack
from infra.monitoring_stack import MonitoringStack
from infra.network_stack import NetworkStack
from infra.security_stack import SecurityStack
from ml_pipeline.config import MODEL_PACKAGE_GROUP_NAME, PROJECT_NAME

app = cdk.App()

env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
)


class MLPlatformStack(cdk.Stack):
    def __init__(self, scope: cdk.App, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        endpoint_name = f"{PROJECT_NAME}-endpoint"

        network = NetworkStack(self, "Network")
        security = SecurityStack(self, "Security")
        data = DataStack(self, "Data", kms_key=security.kms_key)

        deploy = DeployStack(self, "Deploy", endpoint_name=endpoint_name)

        monitoring = MonitoringStack(
            self,
            "Monitoring",
            endpoint_name=endpoint_name,
            lambda_role=security.lambda_role,
            notification_email=self.node.try_get_context("notification_email") or "",
        )

        github_owner = self.node.try_get_context("github_owner")
        github_repo = self.node.try_get_context("github_repo")
        github_connection_arn = self.node.try_get_context("github_connection_arn")

        if github_owner and github_repo and github_connection_arn:
            CicdStack(
                self,
                "Cicd",
                codebuild_role=security.codebuild_role,
                artifacts_bucket=data.artifacts_bucket,
                github_owner=github_owner,
                github_repo=github_repo,
                github_connection_arn=github_connection_arn,
                model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
            )


MLPlatformStack(app, f"{PROJECT_NAME}-stack", env=env)

app.synth()
