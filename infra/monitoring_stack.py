"""
The MLOps feedback loop: SageMaker Model Monitor schedules watch the live
endpoint for data drift and model-quality drift against the baseline
captured during evaluation. When drift crosses a threshold, CloudWatch
alarms fire, SNS notifies, and a Lambda kicks off a new pipeline
execution automatically - closing the loop from "detected drift" back to
"retrained model" without a human needing to notice first.
"""
from aws_cdk import Duration, aws_cloudwatch as cloudwatch, aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_events as events, aws_events_targets as targets
from aws_cdk import aws_iam as iam, aws_lambda as lambda_, aws_sns as sns, aws_sns_subscriptions as subs
from constructs import Construct


class MonitoringStack(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        endpoint_name: str,
        lambda_role: iam.IRole,
        notification_email: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.alerts_topic = sns.Topic(self, "MLAlertsTopic", display_name="ML Pipeline Alerts")
        if notification_email:
            self.alerts_topic.add_subscription(subs.EmailSubscription(notification_email))

        # Model Monitor schedules (data quality + model quality + bias) are
        # created via boto3/CLI once the endpoint and baselines exist,
        # since they need the running endpoint's name and the baseline
        # statistics S3 URIs produced by the Clarify/evaluation steps.
        # See ci_cd/buildspec_deploy.yml for where that call happens.
        # This stack wires the *reaction* to what Model Monitor reports.

        drift_alarm = cloudwatch.Alarm(
            self,
            "ModelQualityDriftAlarm",
            metric=cloudwatch.Metric(
                namespace="aws/sagemaker/Endpoints/model-quality",
                metric_name="auc",
                dimensions_map={"Endpoint": endpoint_name, "MonitoringSchedule": f"{endpoint_name}-quality-monitor"},
                statistic="Average",
                period=Duration.hours(1),
            ),
            threshold=0.75,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD,
        )
        drift_alarm.add_alarm_action(cw_actions.SnsAction(self.alerts_topic))

        self.retrain_trigger_fn = lambda_.Function(
            self,
            "RetrainTriggerFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.handler",
            role=lambda_role,
            timeout=Duration.seconds(30),
            code=lambda_.Code.from_inline(
                """
import boto3

sm = boto3.client("sagemaker")
PIPELINE_NAME = "ml-sagemaker-pipeline-training-pipeline"

def handler(event, context):
    print(f"Drift detected, starting retraining pipeline: {event}")
    response = sm.start_pipeline_execution(PipelineName=PIPELINE_NAME)
    print(f"Started execution: {response['PipelineExecutionArn']}")
    return {"status": "retraining_started", "arn": response["PipelineExecutionArn"]}
"""
            ),
        )
        self.alerts_topic.add_subscription(subs.LambdaSubscription(self.retrain_trigger_fn))

        # Also retrain on a fixed schedule regardless of drift, as a
        # baseline safety net for slow-moving drift the monitor might miss.
        events.Rule(
            self,
            "ScheduledRetrainRule",
            schedule=events.Schedule.rate(Duration.days(7)),
            targets=[targets.LambdaFunction(self.retrain_trigger_fn)],
        )
