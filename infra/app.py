#!/usr/bin/env python3
"""AWS CDK app.

    cdk deploy -c stage=beta -c bedrock_model_id=<bedrock-model-id>
"""

import aws_cdk as cdk

from stack import RootCiteStack

app = cdk.App()
stage = app.node.try_get_context("stage") or "beta"
bedrock_model_id = app.node.try_get_context("bedrock_model_id")
if not bedrock_model_id:
    raise SystemExit("Missing context: pass -c bedrock_model_id=<bedrock-model-id>")

RootCiteStack(app, f"RootCite-{stage}", stage=stage, bedrock_model_id=bedrock_model_id)
app.synth()
