#!/usr/bin/env bash
# Require a pull request + green CI (ruff, bandit, pytest) before anything lands
# on main. Idempotent — run once after the CI workflow exists on GitHub:
#
#   ./scripts/setup-branch-protection.sh [branch]
#
# Needs the GitHub CLI authenticated with repo-admin rights (gh auth login).
set -euo pipefail
cd "$(dirname "$0")/.."

command -v gh >/dev/null 2>&1 || {
  echo "ERROR: GitHub CLI 'gh' not found → brew install gh"; exit 1; }

REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
BRANCH="${1:-main}"

echo "==> Protecting ${REPO}@${BRANCH}: require PR + passing CI (ruff, bandit, pytest)…"
gh api -X PUT "repos/${REPO}/branches/${BRANCH}/protection" \
  --input scripts/branch-protection.json >/dev/null

echo "✅ ${BRANCH} protected:"
echo "   • direct pushes blocked — changes go through a pull request"
echo "   • a PR can only merge when ruff + bandit + pytest are green (and up to date)"
echo "   • applies to admins too (enforce_admins). For an admin escape hatch instead:"
echo "       gh api -X DELETE repos/${REPO}/branches/${BRANCH}/protection/enforce_admins"
