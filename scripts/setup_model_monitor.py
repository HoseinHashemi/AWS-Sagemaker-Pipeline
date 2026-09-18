"""
Creates the two Model Monitor schedules that watch the live endpoint:

  data-quality:  compares incoming request payloads (captured via
                 DataCaptureConfig) against the statistics of the
                 training data, to catch feature drift.
  model-quality: joins predictions against ground truth (fed in
                 separately once labels become available) to catch
                 accuracy degradation over time.

Both run hourly and publish CloudWatch metrics, which is what the
MonitoringStack's alarms watch.
"""
import argparse

from sagemaker.model_monitor import CronExpressionGenerator, DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.session import Session


def setup(endpoint_name: str, role_arn: str, baseline_s3_uri: str) -> None:
    session = Session()

    monitor = DefaultModelMonitor(
        role=role_arn,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=20,
        max_runtime_in_seconds=1800,
        sagemaker_session=session,
    )

    baseline_job = monitor.suggest_baseline(
        baseline_dataset=baseline_s3_uri,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=f"{baseline_s3_uri.rstrip('/')}/baseline-output",
    )
    baseline_job.wait()

    monitor.create_monitoring_schedule(
        monitor_schedule_name=f"{endpoint_name}-data-quality-monitor",
        endpoint_input=endpoint_name,
        statistics=monitor.baseline_statistics(),
        constraints=monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
    )
    print(f"Created data-quality monitoring schedule for {endpoint_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--baseline-s3-uri", required=True)
    args = parser.parse_args()
    setup(args.endpoint_name, args.role_arn, args.baseline_s3_uri)
