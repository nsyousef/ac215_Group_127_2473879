#!/bin/bash

# Script to copy the out/ directory to the packaged Electron app
# This is needed because electron-forge doesn't automatically include it

set -e

# Detect architecture - try arm64 first, then x64
if [ -d "./out/pibu_ai-darwin-arm64/pibu_ai.app/Contents/Resources/app" ]; then
  APP_PATH="./out/pibu_ai-darwin-arm64/pibu_ai.app/Contents/Resources/app"
  APP_DIR="pibu_ai-darwin-arm64"
elif [ -d "./out/pibu_ai-darwin-x64/pibu_ai.app/Contents/Resources/app" ]; then
  APP_PATH="./out/pibu_ai-darwin-x64/pibu_ai.app/Contents/Resources/app"
  APP_DIR="pibu_ai-darwin-x64"
else
  echo "   Error: Packaged app not found"
  echo "   Expected at ./out/pibu_ai-darwin-arm64/ or ./out/pibu_ai-darwin-x64/"
  echo "   Run 'npx electron-forge package' first"
  exit 1
fi

echo "Found app at: $APP_PATH"

if [ -f "./out/index.html" ]; then
  echo "📦 Copying out/ directory (static exports) to packaged app..."

  # Copy only the static files, excluding the packaged app directory itself
  mkdir -p "$APP_PATH/out"

  for item in ./out/*; do
    # Skip the packaged app directories
    basename_item="$(basename "$item")"
    if [ "$basename_item" != "pibu_ai-darwin-arm64" ] && [ "$basename_item" != "pibu_ai-darwin-x64" ]; then
      echo "  Copying $basename_item..."
      cp -r "$item" "$APP_PATH/out/"
    fi
  done

  if [ -f "$APP_PATH/out/index.html" ]; then
    echo "✅ Successfully copied static files to packaged app"
  else
    echo "❌ Failed: index.html not found in copied files"
    exit 1
  fi
else
  echo "⚠️  ./out/index.html not found"
  exit 1
fi
