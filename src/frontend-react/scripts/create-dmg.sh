#!/bin/bash

# Script to create DMG for pibu_ai
# Assumes electron-forge has already packaged the app

set -e

cd "$(dirname "$0")/.."

# Find the packaged app
APP_DIR=$(find ./out -name "pibu_ai-darwin-x64" -type d | head -1)

if [ -z "$APP_DIR" ]; then
  echo "❌ Error: Could not find packaged app in ./out"
  echo "   Run 'npx electron-forge package' first"
  exit 1
fi

echo "📦 Found packaged app at: $APP_DIR"

# Create output directory if it doesn't exist
mkdir -p ./dist

# Create DMG
DMG_FILE="./dist/pibu_ai.dmg"
echo "🎬 Creating DMG: $DMG_FILE"
echo "   Volume name: pibu_ai"

hdiutil create \
  -srcfolder "$APP_DIR" \
  -volname "pibu_ai" \
  -format UDZO \
  "$DMG_FILE"

if [ -f "$DMG_FILE" ]; then
  DMG_SIZE=$(du -h "$DMG_FILE" | cut -f1)
  echo "✅ DMG created successfully!"
  echo "   File: $DMG_FILE"
  echo "   Size: $DMG_SIZE"
  echo ""
  echo "📦 To install:"
  echo "   1. Open the DMG file"
  echo "   2. Drag 'pibu_ai.app' to the Applications folder"
  echo "   3. Launch from Applications"
else
  echo "❌ Error: DMG file was not created"
  exit 1
fi
