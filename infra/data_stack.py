"""
Data layer: the S3 "data lake" buckets and a Glue Data Catalog database
so raw and processed data can be queried with Athena without moving it.

Buckets:
  raw          -> unlabeled/incoming data, written by ingestion jobs
  labeled      -> output manifests from Ground Truth labeling jobs
  processed    -> features produced by SageMaker Processing (train/val/test splits)
  artifacts    -> model artifacts, evaluation reports, pipeline outputs
"""
from aws_cdk import RemovalPolicy, aws_glue as glue, aws_iam as iam, aws_kms as kms, aws_s3 as s3
from constructs import Construct


class DataStack(Construct):
    def __init__(
        self, scope: Construct, construct_id: str, kms_key: kms.IKey, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        common_bucket_kwargs = dict(
            encryption=s3.BucketEncryption.KMS,
            encryption_key=kms_key,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=True,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        self.raw_bucket = s3.Bucket(self, "RawDataBucket", **common_bucket_kwargs)
        self.labeled_bucket = s3.Bucket(self, "LabeledDataBucket", **common_bucket_kwargs)
        self.processed_bucket = s3.Bucket(self, "ProcessedDataBucket", **common_bucket_kwargs)
        self.artifacts_bucket = s3.Bucket(self, "ArtifactsBucket", **common_bucket_kwargs)

        from aws_cdk import Stack

        account = Stack.of(self).account
        self.glue_database = glue.CfnDatabase(
            self,
            "MLGlueDatabase",
            catalog_id=account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name="ml_pipeline_catalog",
                description="Catalog of raw and processed datasets used by the ML pipeline (queryable via Athena)",
            ),
        )

        # Crawler keeps the Glue Catalog schema in sync with what's actually
        # landing in the raw bucket, so Athena queries and the processing
        # job's expected schema don't silently drift apart.
        self.crawler_role = iam.Role(
            self,
            "GlueCrawlerRole",
            assumed_by=iam.ServicePrincipal("glue.amazonaws.com"),
        )
        self.crawler_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSGlueServiceRole")
        )
        self.raw_bucket.grant_read(self.crawler_role)
        kms_key.grant_decrypt(self.crawler_role)

        self.raw_data_crawler = glue.CfnCrawler(
            self,
            "RawDataCrawler",
            role=self.crawler_role.role_arn,
            database_name="ml_pipeline_catalog",
            targets=glue.CfnCrawler.TargetsProperty(
                s3_targets=[glue.CfnCrawler.S3TargetProperty(path=f"s3://{self.raw_bucket.bucket_name}/")]
            ),
            schedule=glue.CfnCrawler.ScheduleProperty(schedule_expression="cron(0 * * * ? *)"),
        )
        self.raw_data_crawler.add_dependency(self.glue_database)
