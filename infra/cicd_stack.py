"""
MLOps glue: two CodePipelines.

  1. build-pipeline: triggers on every push to main. Runs unit tests,
     builds/pushes any custom containers to ECR, then upserts and starts
     the SageMaker Pipeline (preprocess -> tune -> evaluate -> register).
     This is "CI" for ML code.

  2. deploy-pipeline: triggers via EventBridge whenever a model package's
     status changes to "Approved" in the Model Registry (a human approves
     it, or an automated check does). It creates/updates the SageMaker
     endpoint from that model package, then sets up the Model Monitor
     schedules against it. This is "CD" for the model artifact.

Splitting them this way means training a candidate never touches
production traffic, and deploying never silently retrains - each has one
job and one trigger.
"""
from aws_cdk import aws_codebuild as codebuild, aws_codepipeline as codepipeline
from aws_cdk import aws_codepipeline_actions as actions
from aws_cdk import aws_events as events, aws_events_targets as targets
from aws_cdk import aws_iam as iam, aws_s3 as s3
from constructs import Construct


class CicdStack(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        codebuild_role: iam.IRole,
        artifacts_bucket: s3.IBucket,
        github_owner: str,
        github_repo: str,
        github_connection_arn: str,
        model_package_group_name: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        source_output = codepipeline.Artifact("SourceOutput")
        source_action = actions.CodeStarConnectionsSourceAction(
            action_name="GitHub_Source",
            owner=github_owner,
            repo=github_repo,
            branch="main",
            connection_arn=github_connection_arn,
            output=source_output,
        )

        build_project = codebuild.PipelineProject(
            self,
            "MLBuildProject",
            role=codebuild_role,
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                privileged=True,  # needed to build/push container images
            ),
            build_spec=codebuild.BuildSpec.from_source_filename("ci_cd/buildspec_build.yml"),
        )

        self.build_pipeline = codepipeline.Pipeline(
            self,
            "MLBuildPipeline",
            pipeline_name="ml-pipeline-build",
            stages=[
                codepipeline.StageProps(stage_name="Source", actions=[source_action]),
                codepipeline.StageProps(
                    stage_name="BuildAndTrain",
                    actions=[
                        actions.CodeBuildAction(
                            action_name="RunTestsAndTriggerTraining",
                            project=build_project,
                            input=source_output,
                        )
                    ],
                ),
            ],
        )

        deploy_project = codebuild.PipelineProject(
            self,
            "MLDeployProject",
            role=codebuild_role,
            environment=codebuild.BuildEnvironment(build_image=codebuild.LinuxBuildImage.STANDARD_7_0),
            build_spec=codebuild.BuildSpec.from_source_filename("ci_cd/buildspec_deploy.yml"),
        )

        deploy_source_output = codepipeline.Artifact("DeploySourceOutput")
        self.deploy_pipeline = codepipeline.Pipeline(
            self,
            "MLDeployPipeline",
            pipeline_name="ml-pipeline-deploy",
            stages=[
                codepipeline.StageProps(
                    stage_name="Source",
                    actions=[
                        actions.CodeStarConnectionsSourceAction(
                            action_name="GitHub_Source",
                            owner=github_owner,
                            repo=github_repo,
                            branch="main",
                            connection_arn=github_connection_arn,
                            output=deploy_source_output,
                        )
                    ],
                ),
                codepipeline.StageProps(
                    stage_name="ApproveDeployment",
                    actions=[
                        actions.ManualApprovalAction(
                            action_name="ManualApprovalGate",
                            additional_information="Model was auto-approved by CI checks or a human in the Model Registry. Confirm before it goes to the live endpoint.",
                        )
                    ],
                ),
                codepipeline.StageProps(
                    stage_name="Deploy",
                    actions=[
                        actions.CodeBuildAction(
                            action_name="DeployApprovedModel",
                            project=deploy_project,
                            input=deploy_source_output,
                        )
                    ],
                ),
            ],
        )

        # EventBridge rule: SageMaker emits this event automatically when a
        # model package's ModelApprovalStatus changes. We use it to kick
        # off the deploy pipeline instead of polling the registry.
        model_approved_rule = events.Rule(
            self,
            "ModelApprovedRule",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Model Package State Change"],
                detail={
                    "ModelPackageGroupName": [model_package_group_name],
                    "ModelApprovalStatus": ["Approved"],
                },
            ),
        )
        model_approved_rule.add_target(targets.CodePipeline(self.deploy_pipeline))
