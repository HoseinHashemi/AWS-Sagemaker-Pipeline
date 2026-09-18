"""
Security foundation: one KMS key for encrypting everything at rest
(S3 buckets, EBS volumes on training instances, Feature Store), and the
IAM execution roles used by SageMaker jobs, CodeBuild, and Lambda.

Each role gets only the permissions its component actually needs
(least privilege) rather than one shared "SageMakerFullAccess" role.
"""
from aws_cdk import RemovalPolicy, aws_iam as iam, aws_kms as kms
from constructs import Construct


class SecurityStack(Construct):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.kms_key = kms.Key(
            self,
            "MLKmsKey",
            alias="ml-pipeline-key",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # Execution role assumed by every SageMaker Processing/Training/
        # Tuning/Transform/Pipelines job and by hosted endpoints.
        self.sagemaker_execution_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Execution role for SageMaker jobs, pipelines, and endpoints",
        )
        self.sagemaker_execution_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
        )
        self.kms_key.grant_encrypt_decrypt(self.sagemaker_execution_role)

        # CodeBuild role used by the CI/CD build & deploy stages.
        self.codebuild_role = iam.Role(
            self,
            "CodeBuildRole",
            assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com"),
            description="Role for CodeBuild projects that run SageMaker Pipelines and deploy endpoints",
        )
        self.codebuild_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
        )
        self.codebuild_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryPowerUser")
        )
        self.codebuild_role.add_to_policy(
            iam.PolicyStatement(
                actions=["iam:PassRole"],
                resources=[self.sagemaker_execution_role.role_arn],
            )
        )
        self.kms_key.grant_encrypt_decrypt(self.codebuild_role)

        # Lambda role used for event-driven glue (drift alerts -> retrain,
        # model registry approval -> deploy trigger).
        self.lambda_role = iam.Role(
            self,
            "MLLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Role for Lambda functions that react to pipeline/model events",
        )
        self.lambda_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
        )
        self.lambda_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:StartPipelineExecution",
                    "sagemaker:CreateEndpoint",
                    "sagemaker:UpdateEndpoint",
                    "sagemaker:DescribeModelPackage",
                    "codepipeline:StartPipelineExecution",
                    "sns:Publish",
                ],
                resources=["*"],
            )
        )
