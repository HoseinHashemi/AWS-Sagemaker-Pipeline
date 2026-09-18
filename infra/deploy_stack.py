"""
Real-time hosting: a SageMaker endpoint created from whichever model
package in the registry is currently "Approved", fronted by autoscaling
so it scales with traffic instead of running at fixed capacity 24/7.

The endpoint config/model itself is created by the deploy CodeBuild stage
at deploy time (it needs to know the just-approved model package ARN),
so this stack only defines the pieces that are stable ahead of time:
the autoscaling target/policy attached to whatever endpoint name the
deploy stage produces, and the CloudWatch alarms that watch it.
"""
from aws_cdk import Duration, aws_applicationautoscaling as appscaling, aws_cloudwatch as cloudwatch
from constructs import Construct


class DeployStack(Construct):
    def __init__(
        self, scope: Construct, construct_id: str, endpoint_name: str, variant_name: str = "AllTraffic", **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"

        scalable_target = appscaling.ScalableTarget(
            self,
            "EndpointScalableTarget",
            service_namespace=appscaling.ServiceNamespace.SAGEMAKER,
            resource_id=resource_id,
            scalable_dimension="sagemaker:variant:DesiredInstanceCount",
            min_capacity=1,
            max_capacity=4,
        )

        scalable_target.scale_to_track_metric(
            "InvocationsPerInstanceScaling",
            custom_metric=cloudwatch.Metric(
                namespace="AWS/SageMaker",
                metric_name="InvocationsPerInstance",
                dimensions_map={"EndpointName": endpoint_name, "VariantName": variant_name},
                statistic="Sum",
                period=Duration.minutes(1),
            ),
            target_value=1000,
            scale_in_cooldown=Duration.minutes(5),
            scale_out_cooldown=Duration.minutes(1),
        )

        self.latency_alarm = cloudwatch.Alarm(
            self,
            "HighLatencyAlarm",
            metric=cloudwatch.Metric(
                namespace="AWS/SageMaker",
                metric_name="ModelLatency",
                dimensions_map={"EndpointName": endpoint_name, "VariantName": variant_name},
                statistic="Average",
                period=Duration.minutes(5),
            ),
            threshold=1_000_000,  # microseconds = 1s
            evaluation_periods=3,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
        )

        self.error_alarm = cloudwatch.Alarm(
            self,
            "InvocationErrorAlarm",
            metric=cloudwatch.Metric(
                namespace="AWS/SageMaker",
                metric_name="Invocation4XXErrors",
                dimensions_map={"EndpointName": endpoint_name, "VariantName": variant_name},
                statistic="Sum",
                period=Duration.minutes(5),
            ),
            threshold=10,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
        )
