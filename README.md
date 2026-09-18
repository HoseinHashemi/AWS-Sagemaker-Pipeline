# End-to-End ML Pipeline on Amazon SageMaker

A production-shaped, end-to-end machine learning platform built on SageMaker
and the AWS services around it. Every stage of the ML lifecycle is here:
data storage and cataloging, annotation, feature engineering, training with
hyperparameter tuning, evaluation with a hard quality gate, bias checking,
model governance, CI/CD, real-time + batch deployment, and drift monitoring
that closes the loop back into retraining.

This project is intentionally built to *touch a lot of AWS services* — it
doubles as interview prep material for talking through a real ML platform
architecture.

## The one-paragraph version

New data lands in S3 → gets labeled by Ground Truth if needed → a
**SageMaker Pipeline** (the orchestrator) preprocesses it, tunes a model,
evaluates it against a held-out test set, and only registers it in the
**Model Registry** if it clears a quality bar. A human (or an automated
check) approves the model package, which fires an **EventBridge** event
that triggers a **CodePipeline** deploy stage. That stage stands up a
**SageMaker endpoint** behind autoscaling, and wires up **Model Monitor**
to watch it. When Model Monitor sees drift, it alerts via **SNS**, and a
**Lambda** kicks off a new pipeline execution automatically — closing the
loop.

## Architecture diagram

```
 ┌─────────────┐     ┌──────────────┐     ┌───────────────────┐
 │   Raw data   │────▶│Ground Truth  │────▶│   S3 (labeled)     │
 │  S3 bucket   │     │ labeling job │     │                    │
 └─────────────┘     └──────────────┘     └─────────┬──────────┘
                                                      │
                                    Glue Crawler / Athena (ad-hoc queries)
                                                      │
                                                      ▼
                          ┌───────────────────────────────────────────┐
                          │           SageMaker Pipeline (DAG)         │
                          │                                             │
                          │  Preprocess ──▶ Tune (HPO) ──▶ Evaluate     │
                          │  (Processing)   (Training x N)  (Processing)│
                          │                                    │        │
                          │                                    ▼        │
                          │                          ConditionStep      │
                          │                       (AUC >= threshold?)   │
                          │                            │         │      │
                          │                          yes│         │no    │
                          │                            ▼         ▼      │
                          │                   ClarifyCheck   (stop)     │
                          │                         │                   │
                          │                         ▼                   │
                          │                  RegisterModel               │
                          │              (Model Registry, Pending)       │
                          └──────────────────────┬──────────────────────┘
                                                  │
                                    Manual/automated approval
                                                  │
                                    EventBridge: "Model Approved"
                                                  │
                                                  ▼
                          ┌───────────────────────────────────────────┐
                          │       CodePipeline: deploy stage           │
                          │   create/update SageMaker endpoint         │
                          │   (blue/green), attach autoscaling,        │
                          │   create Model Monitor schedules           │
                          └──────────────────────┬──────────────────────┘
                                                  │
                                                  ▼
                          ┌───────────────────────────────────────────┐
                          │   SageMaker real-time endpoint (+ batch    │
                          │   transform for offline scoring)           │
                          │   CloudWatch alarms + Model Monitor        │
                          └──────────────────────┬──────────────────────┘
                                                  │
                                       drift detected → SNS → Lambda
                                                  │
                                                  ▼
                                  starts a new SageMaker Pipeline execution
                                          (back to the top)
```

## Why this shape

**Everything is one DAG, not a pile of scripts.** SageMaker Pipelines
gives you a single definition (`ml_pipeline/pipeline.py`) that AWS executes,
retries, caches, and tracks lineage for. In an interview, this is the
answer to "how do you avoid a Jupyter notebook full of steps nobody can
re-run" — you commit the DAG to git and it's now infrastructure.

**A quality gate, not just automation.** The `ConditionStep` means a model
that underperforms simply never reaches the registry. That's the difference
between "we ran the pipeline" and "we have a production gate." This is one
of the highest-value things to be able to explain clearly.

**Registry + approval + event, not a script that deploys on every run.**
Training and deployment are two separate CodePipelines with two separate
triggers (git push vs. model approval). This means retraining never touches
production by accident, and a bad approval can still be caught by a manual
gate before the endpoint changes.

**Monitor → alert → retrain is a closed loop.** A model that's never
re-evaluated after deployment is a liability. Model Monitor + CloudWatch +
SNS + Lambda is what turns "we deployed a model" into "we have a system
that notices when the model goes stale and fixes itself."

**Least-privilege IAM roles per component**, not one shared
`SageMakerFullAccess`-everywhere role — SageMaker jobs, CodeBuild, and
Lambda each get their own role scoped to what they actually do.

**No NAT gateway.** SageMaker jobs run in private subnets with VPC
endpoints for S3/ECR/SageMaker/CloudWatch/STS instead of routing through a
NAT gateway to the internet — cheaper and it also means the jobs have no
path to the public internet at all, which is a security property worth
naming out loud.

## Walking through the ML lifecycle stage by stage

### 1. Data & annotation
- **S3** (`infra/data_stack.py`): four buckets — raw, labeled, processed,
  artifacts — each KMS-encrypted, versioned, no public access.
- **Glue Crawler + Data Catalog**: keeps a live schema of what's actually
  in the raw bucket, so **Athena** can run ad-hoc SQL against it without
  anyone having to remember the schema by hand.
- **SageMaker Ground Truth** (`src/labeling/create_labeling_job.py`):
  spins up a labeling job against a private workforce for unlabeled data.
  This is a manual trigger, not part of the automated pipeline — labeling
  needs a human decision about when a batch is ready, unlike training.

### 2. Feature engineering
- **SageMaker Processing** (`src/processing/preprocess.py`): cleans nulls,
  does the train/validation/test split (stratified, so class balance is
  preserved in every split), writes back to S3.
- (Design note, not built here to keep scope sane: a **SageMaker Feature
  Store** would sit here in a bigger system, so the same feature
  definitions are reused consistently between training and low-latency
  online inference instead of being recomputed differently in each place.)

### 3. Training & tuning
- **SageMaker Training Jobs** run inside a **HyperparameterTuner**
  (`ml_pipeline/pipeline.py`, step "TuneModel") — Bayesian search over
  learning rate and regularization, several jobs in parallel, Spot
  instances for cost. This is "Automatic Model Tuning" in AWS terms.
- The training script (`src/training/train.py`) prints
  `validation-auc: <value>` in the exact format SageMaker's metric-regex
  parser expects, which is how the tuner knows which job won.

### 4. Evaluation & quality gate
- **SageMaker Processing** again (`src/processing/evaluate.py`) scores the
  winning model against the untouched test split and writes
  `evaluation.json` in the SageMaker "model quality report" shape.
- The pipeline's `ConditionStep` reads that file's AUC value with
  `JsonGet` and compares it against a pipeline parameter
  (`MinModelAUC`, default from `ml_pipeline/config.py`). Below the bar,
  the pipeline execution ends there — nothing gets registered.

### 5. Bias & explainability
- **SageMaker Clarify** (`ClarifyCheckStep` in `pipeline.py`) computes a
  bias baseline (label imbalance across a sensitive feature) on the
  training data. This baseline is what a later Model Monitor bias-drift
  schedule would compare live traffic against.

### 6. Model governance
- **SageMaker Model Registry**: candidate models land here as model
  package versions with `PendingManualApproval` status, carrying their
  evaluation metrics attached (`ModelMetrics`) — anyone reviewing can see
  the AUC/precision/recall right next to the model, not in a separate doc.
- SageMaker automatically tracks **lineage** (which data, which code
  commit conceptually, which training job produced which model) — worth
  mentioning by name even though there's no separate resource for it.

### 7. CI/CD (MLOps)
- **CodePipeline #1 — build** (`infra/cicd_stack.py`, `ci_cd/buildspec_build.yml`):
  triggers on push to `main`. Runs unit tests, then upserts and starts the
  SageMaker Pipeline. This is CI for ML code — it produces a *candidate*.
- **CodePipeline #2 — deploy** (`ci_cd/buildspec_deploy.yml`): triggered
  by an **EventBridge** rule matching SageMaker's own
  "Model Package State Change" event when a model is Approved. Has a
  manual approval action before it touches the live endpoint. This is CD
  for the model *artifact* — deliberately decoupled from CI so a
  retraining run can never silently redeploy.

### 8. Deployment
- **Real-time**: `scripts/deploy_approved_model.py` grabs the latest
  Approved model package and creates/updates a **SageMaker endpoint**.
  Because it re-uses the same endpoint name, `update_endpoint` triggers
  SageMaker's built-in blue/green rollout — new variant health-checked
  before traffic shifts, old variant torn down only after.
- **Autoscaling** (`infra/deploy_stack.py`): Application Auto Scaling
  target tracking on `InvocationsPerInstance`, so instance count follows
  traffic instead of running fixed capacity.
- **Data capture** is turned on for the endpoint, which is what feeds
  Model Monitor real traffic to compare against the baseline.
- **Batch Transform** is the other deployment mode mentioned in the
  design for offline/bulk scoring jobs that don't need a live endpoint.

### 9. Monitoring & the feedback loop
- **SageMaker Model Monitor** (`scripts/setup_model_monitor.py`): hourly
  data-quality checks comparing live captured traffic against the
  training baseline statistics.
- **CloudWatch alarms** (`infra/deploy_stack.py`, `infra/monitoring_stack.py`)
  watch endpoint latency, 4xx errors, and the model-quality metric Model
  Monitor publishes.
- **SNS + Lambda** (`infra/monitoring_stack.py`): an alarm firing notifies
  an SNS topic, which triggers a Lambda that calls
  `start_pipeline_execution` — a new candidate model starts training
  automatically. A weekly EventBridge schedule does the same as a
  safety net for slow drift the monitor might miss.

### 10. Security posture
- One **KMS** customer-managed key encrypts everything at rest — S3,
  Feature Store, EBS volumes on training instances.
- Three IAM roles, each scoped to one component (SageMaker jobs,
  CodeBuild, Lambda) instead of one shared broad role.
- **VPC** with only private isolated subnets and interface/gateway VPC
  endpoints — SageMaker jobs never need a route to the public internet.

## Repo layout

```
app.py                     CDK app entry point — wires all infra stacks together
infra/                      CDK constructs: network, security, data, deploy, monitoring, CI/CD
ml_pipeline/pipeline.py     The SageMaker Pipeline DAG definition (the core artifact)
ml_pipeline/config.py       Shared names/thresholds (model package group, quality bar)
src/processing/             Preprocessing + evaluation scripts (run inside Processing jobs)
src/training/train.py       Training entry point (runs inside Training jobs)
src/inference/inference.py  Custom model_fn/input_fn/predict_fn/output_fn for the endpoint
src/labeling/                Ground Truth labeling job launcher
scripts/                    One-off deploy-time scripts (deploy model, set up Model Monitor)
ci_cd/                      buildspecs for the two CodePipelines
tests/                      Unit tests for the preprocessing/training/inference logic
```

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# unit tests
pytest tests/ -v

# see the generated CloudFormation without deploying anything
cdk synth --context github_owner=<you> --context github_repo=<repo> \
  --context github_connection_arn=<arn>

# deploy the platform (needs real AWS credentials + a GitHub CodeStar connection)
cdk deploy --context github_owner=<you> --context github_repo=<repo> \
  --context github_connection_arn=<arn> --context notification_email=<email>

# build/inspect the SageMaker Pipeline definition itself (needs AWS creds + a real S3 bucket)
python ml_pipeline/pipeline.py --role-arn <sagemaker-execution-role-arn> \
  --artifacts-bucket <bucket> --region us-east-1 --upsert --start
```

## What's deliberately out of scope

To keep this a learnable, reviewable system rather than a sprawl:
Feature Store, a real GitHub CodeStar Connection, and a production Ground
Truth workforce are referenced in the design and code comments but not
fully wired up end-to-end — they're the natural "next things to add" if
you want to keep extending this for practice.
