#!/bin/bash

echo "Fetching all your repositories..."

# Get all repos (owned + org + collaborator)
ALL_REPOS=$(gh api user/repos --paginate --jq '.[] | select(.permissions.push == true) | .full_name' 2>&1)

if [ -z "$ALL_REPOS" ]; then
  echo "❌ No repositories found"
  exit 1
fi

echo "Found $(echo "$ALL_REPOS" | wc -l | xargs) repositories with write access"
echo ""

for REPO in $ALL_REPOS; do
  [ -z "$REPO" ] && continue

  echo "================================"
  echo "Processing: $REPO"

  # Get ALL artifacts (including expired ones!)
  ARTIFACT_IDS=$(gh api "repos/$REPO/actions/artifacts" --paginate --jq '.artifacts[]? | .id' 2>/dev/null)

  if [ $? -ne 0 ]; then
    echo "⚠️  Cannot access artifacts"
    continue
  fi

  if [ -z "$ARTIFACT_IDS" ]; then
    echo "✅ No artifacts"
  else
    COUNT=$(echo "$ARTIFACT_IDS" | wc -l | xargs)
    echo "🗑️  Deleting $COUNT artifact(s) (including expired)..."

    echo "$ARTIFACT_IDS" | while read -r ID; do
      gh api -X DELETE "repos/$REPO/actions/artifacts/$ID" 2>/dev/null && echo "  ✓ Deleted $ID"
    done
  fi
done

echo "================================"
