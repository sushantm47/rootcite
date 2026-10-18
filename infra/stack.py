"""RootCite on AWS: ECS Fargate behind an ALB, RDS PostgreSQL with pgvector, Claude on Bedrock.

* No API keys: the task role calls Bedrock with IAM (least privilege: InvokeModel only).
* Database credentials live in Secrets Manager and are injected at runtime.
* Alarms on p99 latency and 5xx follow the service's SLOs (see docs/DESIGN.md).
"""

from aws_cdk import Duration, RemovalPolicy, Stack
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_ecs_patterns as ecs_patterns
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_rds as rds
from constructs import Construct


class RootCiteStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, *, stage: str, bedrock_model_id: str, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        is_prod = stage == "prod"

        vpc = ec2.Vpc(self, "Vpc", max_azs=2, nat_gateways=1)

        db = rds.DatabaseInstance(
            self,
            "Db",
            engine=rds.DatabaseInstanceEngine.postgres(version=rds.PostgresEngineVersion.VER_16),
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.BURSTABLE4_GRAVITON, ec2.InstanceSize.MEDIUM
            ),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            credentials=rds.Credentials.from_generated_secret("rootcite"),
            database_name="rootcite",
            multi_az=is_prod,
            storage_encrypted=True,
            backup_retention=Duration.days(7 if is_prod else 1),
            removal_policy=RemovalPolicy.SNAPSHOT if is_prod else RemovalPolicy.DESTROY,
        )
        assert db.secret is not None

        cluster = ecs.Cluster(self, "Cluster", vpc=vpc, container_insights=True)
        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "Api",
            cluster=cluster,
            cpu=1024,
            memory_limit_mib=2048,
            desired_count=2 if is_prod else 1,
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_asset(".."),
                container_port=8000,
                environment={
                    "ROOTCITE_STAGE": stage,
                    "ROOTCITE_LLM_PROVIDER": "bedrock",
                    "ROOTCITE_MODEL": bedrock_model_id,
                    "ROOTCITE_INDEX": "postgres",
                    "AWS_REGION": self.region,
                    "DB_HOST": db.db_instance_endpoint_address,
                    "DB_NAME": "rootcite",
                },
                secrets={
                    "DB_USER": ecs.Secret.from_secrets_manager(db.secret, "username"),
                    "DB_PASSWORD": ecs.Secret.from_secrets_manager(db.secret, "password"),
                },
            ),
        )
        service.target_group.configure_health_check(path="/health")
        db.connections.allow_default_port_from(service.service)

        service.task_definition.task_role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=["*"],  # narrow to the model/inference-profile ARN per account
            )
        )

        scaling = service.service.auto_scale_task_count(min_capacity=1, max_capacity=6)
        scaling.scale_on_cpu_utilization("Cpu", target_utilization_percent=60)

        service.target_group.metrics.target_response_time(
            statistic="p99", period=Duration.minutes(1)
        ).create_alarm(
            self,
            "P99LatencyAlarm",
            threshold=20,  # seconds; SLO is p99 < 20s for a full investigation
            evaluation_periods=5,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        service.load_balancer.metrics.http_code_target(
            code=elbv2.HttpCodeTarget.TARGET_5XX_COUNT, period=Duration.minutes(1)
        ).create_alarm(
            self,
            "Target5xxAlarm",
            threshold=5,
            evaluation_periods=3,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
