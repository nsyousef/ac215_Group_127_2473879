#!/bin/bash

# Script to create DMG for pibu_ai with drag-to-install interface
# Shows: pibu_ai.app -> Applications symlink for easy installation

set -e

cd "$(dirname "$0")/.."

# Find the packaged app - try arm64 first, then x64
APP_DIR=$(find ./out -name "pibu_ai-darwin-arm64" -type d 2>/dev/null | head -1)
if [ -z "$APP_DIR" ]; then
  APP_DIR=$(find ./out -name "pibu_ai-darwin-x64" -type d 2>/dev/null | head -1)
fi

if [ -z "$APP_DIR" ]; then
  echo "❌ Error: Could not find packaged app in ./out"
  echo "   Expected pibu_ai-darwin-arm64 or pibu_ai-darwin-x64"
  echo "   Run 'npx electron-forge package' first"
  exit 1
fi

echo "📦 Found packaged app at: $APP_DIR"

# Create output directory if it doesn't exist
mkdir -p ./dist

# Create temporary DMG staging folder
STAGING_DIR="./dist/pibu_ai_dmg_staging"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

echo "🎬 Setting up DMG layout..."

# Copy the app into staging folder
# macOS adds com.apple.provenance extended attributes to many files in the
# bundled Python environment. Attempting to copy those attributes inside the
# Codex sandbox triggers "Operation not permitted", which then makes cp report
# dozens of "No such file or directory" errors for the Python packages (torch,
# sympy, scipy, …). Using -X tells BSD cp to skip extended attributes/ACLs so
# the filesystem tree can be copied cleanly.
cp -R -X "$APP_DIR/pibu_ai.app" "$STAGING_DIR/pibu_ai.app"

# Create symbolic link to Applications folder
ln -s /Applications "$STAGING_DIR/Applications"

# Create DMG from staging folder
DMG_FILE="./dist/pibu_ai.dmg"
echo "🎬 Creating DMG: $DMG_FILE"

hdiutil create \
  -srcfolder "$STAGING_DIR" \
  -volname "pibu_ai" \
  -format UDZO \
  -imagekey zlib-level=9 \
  "$DMG_FILE"

# Clean up staging folder
rm -rf "$STAGING_DIR"

if [ -f "$DMG_FILE" ]; then
  DMG_SIZE=$(du -h "$DMG_FILE" | cut -f1)
  echo "✅ DMG created successfully!"
  echo "   File: $DMG_FILE"
  echo "   Size: $DMG_SIZE"
  echo ""
  echo "📦 Installation instructions:"
  echo "   1. Open pibu_ai.dmg"
  echo "   2. Drag pibu_ai.app to Applications (shown in the DMG window)"
  echo "   3. Launch from Applications"
else
  echo "❌ Error: DMG file was not created"
  exit 1
fi
