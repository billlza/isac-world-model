#!/usr/bin/env bash
# Creates a NEW private repository only. Never force-pushes or alters visibility.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-billlza/isac-world-model}"
case "$TARGET" in billlza/*) ;; *) echo "Refusing: expected billlza/<new-name>" >&2; exit 2;; esac
[[ "$TARGET" =~ ^billlza/[A-Za-z0-9._-]+$ ]] || { echo 'Invalid repository name' >&2; exit 2; }
command -v gh >/dev/null || { echo 'Install GitHub CLI first, e.g. brew install gh on macOS.' >&2; exit 2; }
command -v git >/dev/null || { echo 'git is required.' >&2; exit 2; }
gh auth status --hostname github.com >/dev/null
LOGIN="$(gh api user --jq .login)"
[[ "$LOGIN" == billlza ]] || { echo "Wrong GitHub account: $LOGIN" >&2; exit 2; }
if gh repo view "$TARGET" >/dev/null 2>&1; then
  echo "Refusing to modify existing repository $TARGET. Choose a new repository name." >&2
  exit 2
fi
cd "$ROOT"
if [[ -d .git ]]; then
  if git remote get-url origin >/dev/null 2>&1; then
    echo 'An origin already exists; inspect it instead of auto-publishing.' >&2; exit 2
  fi
  [[ -z "$(git status --porcelain)" ]] || { echo 'Working tree is dirty; inspect and commit it first.' >&2; exit 2; }
else
  git init -b main
  # Limit staging to the delivered source/doc paths; never stage local data or secrets.
  git add README.md CHANGELOG.md RIGHTS.md AGENTS.md pyproject.toml Cargo.toml rust-toolchain.toml MANIFEST.in .gitignore src tests crates scripts ci docs examples validation .github
  [[ ! -f Cargo.lock ]] || git add Cargo.lock
  if ! git var GIT_AUTHOR_IDENT >/dev/null 2>&1; then
    echo 'Set git user.name and user.email, then rerun. No remote was created.' >&2; exit 2
  fi
  git commit -m 'feat: independently implemented from-scratch ISAC world-model prototype'
fi
# --private is deliberate. The CLI will use the user's existing authentication.
gh repo create "$TARGET" --private --source . --remote origin \
  --description 'From-scratch visual-radio world model: causal asynchronous research prototype' --push
printf '\nPublished commit: '
git rev-parse HEAD
gh repo view "$TARGET" --json url,visibility --jq '"\(.url) [\(.visibility)]"'
