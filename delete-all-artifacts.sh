#!/bin/bash
set -euo pipefail

KEEP_COUNT=10
REPO=""

usage() {
  cat <<'EOF'
delete-all-artifacts.sh [--repo owner/name] [--keep N]

Deletes GitHub Actions artifacts for a single repository, removing the oldest
artifacts first until only N remain (default: 10). Set --keep 0 to delete all.
Requires the GitHub CLI (`gh`) to be authenticated (GITHUB_TOKEN works).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO="$2"
      shift 2
      ;;
    --keep)
      KEEP_COUNT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI (gh) is required."
  exit 1
fi

if [[ -z "$REPO" ]]; then
  REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
fi

if [[ -z "$REPO" ]]; then
  echo "ERROR: Unable to determine repository."
  exit 1
fi

if [[ "$KEEP_COUNT" =~ [^0-9] ]]; then
  echo "--keep must be a non-negative integer." >&2
  exit 1
fi

echo "Managing artifacts for $REPO (keeping latest $KEEP_COUNT)"

ARTIFACT_LINES=$(gh api "repos/$REPO/actions/artifacts" --paginate \
  --jq '.artifacts[]? | "\(.created_at)\t\(.id)\t\(.name)\t\(.expired)"' | sort)

if [[ -z "$ARTIFACT_LINES" ]]; then
  echo "No artifacts to delete."
  exit 0
fi

TOTAL=$(echo "$ARTIFACT_LINES" | wc -l | xargs)
echo "Found $TOTAL artifact(s)."

KEEP_INT=$((KEEP_COUNT))
if [[ "$KEEP_INT" -ge "$TOTAL" ]]; then
  echo "Nothing to delete (total <= keep)."
  exit 0
fi

DELETE_COUNT=$((TOTAL - KEEP_INT))
[[ "$KEEP_INT" -eq 0 ]] && DELETE_COUNT=$TOTAL

echo "Deleting $DELETE_COUNT oldest artifact(s)..."

echo "$ARTIFACT_LINES" | head -n "$DELETE_COUNT" | while IFS=$'\t' read -r CREATED ID NAME EXPIRED; do
  STATE=$([[ "$EXPIRED" == "true" ]] && echo "expired" || echo "active")
  echo "  • $NAME ($ID) created $CREATED [$STATE]"
  if gh api -X DELETE "repos/$REPO/actions/artifacts/$ID" >/dev/null; then
    echo "    deleted"
  else
    echo "    failed to delete"
  fi
done

echo "Cleanup complete."
