#!/bin/bash

# Script to copy the out/ directory to the packaged Electron app
# This is needed because electron-forge doesn't automatically include it

set -e

APP_PATH="./out/pibu_ai-darwin-x64/pibu_ai.app/Contents/Resources/app"

if [ ! -d "$APP_PATH" ]; then
  echo "❌ Error: App not found at $APP_PATH"
  echo "   Run 'npx electron-forge package' first"
  exit 1
fi

if [ -f "./out/index.html" ]; then
  echo "📦 Copying out/ directory (static exports) to packaged app..."

  # Copy only the static files, excluding the packaged app directory itself
  mkdir -p "$APP_PATH/out"

  for item in ./out/*; do
    # Skip the packaged app directory
    if [ "$(basename "$item")" != "pibu_ai-darwin-x64" ]; then
      echo "  Copying $(basename "$item")..."
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
