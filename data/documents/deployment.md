# Deployment Procedures

## Environments

- `dev` — deploys automatically on merge to `main`.
- `staging` — deploys automatically on merge to `release/*` branches.
- `production` — requires manual approval from a team lead.

## Standard Deployment Flow

1. Open a pull request against `main`.
2. Ensure CI passes (lint, unit tests, integration tests).
3. Get at least one approving review.
4. Merge — this triggers an automatic `dev` deploy.
5. Promote to `staging` by cutting a `release/*` branch.
6. After staging validation, request production approval from a team
   lead in `#deploys`.
7. Production deploys are executed by the on-call deployer, never by the
   author of the change.

## Rollbacks

Use the internal `deploy rollback <service> <version>` command. Rollbacks
do not require additional approval but must be announced in `#deploys`.

## Feature Flags

New risky behavior should ship behind a feature flag rather than a direct
release. Flags are managed through the internal flag service; toggling a
flag in production requires the same approval as a production deploy.

## Secrets in Deployments

Deployment configuration never contains raw secrets. All secrets are
injected at runtime from the internal secrets manager and are never
written to logs, deploy manifests, or documentation.
