"""
Networking foundation: a VPC with private subnets for SageMaker jobs and
endpoints, plus VPC endpoints so traffic to S3/ECR/CloudWatch/SageMaker
never has to leave the AWS network. No NAT gateway is created (saves cost);
everything the training/processing jobs need is reachable via VPC endpoints.
"""
from aws_cdk import aws_ec2 as ec2
from constructs import Construct


class NetworkStack(Construct):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.vpc = ec2.Vpc(
            self,
            "MLVpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="private-isolated",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24,
                ),
            ],
        )

        self.sagemaker_sg = ec2.SecurityGroup(
            self,
            "SageMakerSG",
            vpc=self.vpc,
            description="Security group for SageMaker jobs/endpoints",
            allow_all_outbound=True,
        )

        gateway_services = {
            "S3": ec2.GatewayVpcEndpointAwsService.S3,
        }
        for name, service in gateway_services.items():
            self.vpc.add_gateway_endpoint(f"{name}Endpoint", service=service)

        interface_services = {
            "SageMakerApi": ec2.InterfaceVpcEndpointAwsService.SAGEMAKER_API,
            "SageMakerRuntime": ec2.InterfaceVpcEndpointAwsService.SAGEMAKER_RUNTIME,
            "Ecr": ec2.InterfaceVpcEndpointAwsService.ECR,
            "EcrDocker": ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER,
            "CloudWatchLogs": ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
            "CloudWatchMonitoring": ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH,
            "Sts": ec2.InterfaceVpcEndpointAwsService.STS,
        }
        for name, service in interface_services.items():
            self.vpc.add_interface_endpoint(
                f"{name}Endpoint",
                service=service,
                security_groups=[self.sagemaker_sg],
                private_dns_enabled=True,
            )
