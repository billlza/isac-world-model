#!/usr/bin/env bash
# Explicitly dispatch and wait for the workflow on the exact local commit.
# Does not push, change visibility, or waive a failed/cancelled job.
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET="${1:-billlza/isac-world-model}"
REF="${2:-main}"
[[ "$TARGET" =~ ^billlza/[A-Za-z0-9._-]+$ ]] || { echo 'Unexpected repository' >&2; exit 2; }
[[ "$REF" =~ ^[A-Za-z0-9._/-]+$ ]] || { echo 'Invalid branch' >&2; exit 2; }
command -v gh >/dev/null || { echo 'gh CLI is required' >&2; exit 2; }
[[ "$(gh api user --jq .login)" == billlza ]] || { echo 'Wrong GitHub account' >&2; exit 2; }
[[ "$(gh repo view "$TARGET" --json visibility --jq .visibility)" == PRIVATE ]] || { echo 'Expected PRIVATE repository' >&2; exit 2; }
HEAD_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(gh api "repos/$TARGET/commits/$REF" --jq .sha)"
[[ "$HEAD_SHA" == "$REMOTE_SHA" ]] || { echo 'Remote branch differs from local commit; push deliberately before CI' >&2; exit 2; }
START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
gh workflow run tests.yml --repo "$TARGET" --ref "$REF"
RUN_ID=''
for _ in $(seq 1 30); do
  RUN_ID="$(gh run list --repo "$TARGET" --workflow tests.yml --event workflow_dispatch --limit 20 \
    --json databaseId,headSha,createdAt --jq ".[] | select(.headSha == \"$HEAD_SHA\" and .createdAt >= \"$START\") | .databaseId" | head -n 1)"
  [[ -z "$RUN_ID" ]] || break
  sleep 2
done
[[ -n "$RUN_ID" ]] || { echo 'No matching workflow run found; do not assume success' >&2; exit 2; }
gh run watch "$RUN_ID" --repo "$TARGET" --exit-status --interval 10
gh run view "$RUN_ID" --repo "$TARGET" --json url,headSha,conclusion,jobs
