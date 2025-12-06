# Building pibu.ai for macOS

Complete guide for building the pibu.ai desktop application for macOS with bundled Python.

**Naming Convention:**
- **User-facing name**: `pibu.ai` (shown in Dock, window titles, menus, Applications folder)
- **Filename/internal name**: `pibu_ai` (used in DMG files, app bundles, directories, config files)

**Table of Contents:**
- [Quick Start (TL;DR)](#quick-start-tldr)
- [Prerequisites](#prerequisites)
- [Development Setup](#development-setup)
- [Building the Production App](#building-the-production-app)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

---

## Quick Start (TL;DR)

### For Development (Live Reload)

```bash
cd src/frontend-react
npm install
npm run dev-electron
```

### For Production Build (Mac, works for both Intel and M-series)

```bash
cd src/frontend-react
npm install
npm run bundle-python
npm run build
npm run make-dmg
```

**Output:** `dist/pibu_ai.dmg`

The DMG will have a drag-to-install interface with the app and an Applications folder symlink for easy installation.

### Installing from DMG

1. Open the DMG file
2. Drag `pibu_ai.app` to the `Applications` folder (shown in the DMG window)
3. Launch the app from Applications or use Spotlight Search

---

## Prerequisites

- **Node.js** 16+ and npm 8+
- **Python** 3.11+ (for bundling)
- **Xcode Command Line Tools** (optional, for code signing):
  ```bash
  xcode-select --install
  ```

---

## Development Setup

### 1. Install Dependencies

```bash
cd src/frontend-react
npm install
```

### 2. Bundle Python (First Time Only)

Create a local Python virtual environment with all dependencies:

```bash
npm run bundle-python
```

This creates `resources/python-bundle/venv` with all Python dependencies.

### 3. Run in Development

```bash
npm run dev-electron
```

This:
- Starts Next.js dev server on port 3000
- Launches Electron with live reload
- Auto-reloads on code changes
- Uses development bundled Python

**Development Workflow:**
- Edit React components in `src/components/` → auto-reload in Electron
- Edit Python code in `python/` → restart Python server (Ctrl+C, then `npm run dev-electron`)
- Edit `electron/main.js` → restart Electron (Ctrl+C, then `npm run dev-electron`)

---

## Building the Production App

### Step 1: Install Dependencies

```bash
cd src/frontend-react
npm install
```

### Step 2: Bundle Python

```bash
npm run bundle-python
```

This creates `resources/python-bundle/venv` with all Python dependencies for production.

### Step 3: Build Next.js

```bash
npm run build
```

Compiles the React/Next.js frontend to static assets.

### Step 4: Create DMG Installer

```bash
npm run make-dmg
```

Creates the macOS DMG installer file using `electron-forge` for packaging and `hdiutil` for DMG creation.

**Output:** `dist/pibu_ai.dmg` (drag-to-install format with Applications symlink)

#### What Happens:
1. Builds Next.js frontend (if not already built)
2. Bundles Python virtual environment with all dependencies
3. Packages Electron app using electron-forge
4. Creates a temporary staging folder with:
   - `pibu_ai.app` (the application)
   - `Applications` (symbolic link to /Applications)
5. Creates a compressed, user-friendly DMG file

#### DMG Format:
The generated DMG uses the standard macOS drag-to-install interface that users are familiar with:
- Opens with a Finder window showing two items
- Left side: `pibu_ai.app` (the application to drag)
- Right side: `Applications` folder (destination to drag to)
- Users simply drag the app to Applications to install

### One-Command Build

```bash
npm install && npm run bundle-python && npm run build && npm run make-dmg
```

This runs all steps in sequence and produces the final DMG installer.

---

## Troubleshooting

### Build Issues

The app DMG includes:
- **React/Next.js frontend** - compiled to static assets
- **Python virtual environment** - complete with all dependencies from `python/requirements.txt`
- **Electron runtime** - Chromium browser engine for the desktop app
- **App metadata and permissions** - defined in `build-resources/entitlements.plist`

**User Experience:**
- Users download the DMG and drag pibu.ai to Applications
- No need to install Python, Node.js, or npm
- App works completely standalone
- User data stored in `~/Library/Application Support/pibu_ai/`

### Verification

Test the built app:

```bash
# Open the DMG installer
open out/make/Derma\ Assistant-*.dmg

# Or directly run the app
open out/make/Derma\ Assistant-x64-*/Derma\ Assistant.app
```

---

## File Structure

```
src/frontend-react/
├── scripts/
│   ├── bundle-python-macos.sh    # Python bundling script
│   ├── generate-icon.html
│   └── setup-python-env.js
├── build-resources/
│   ├── icon.png                  # App icon (1024x1024)
│   ├── icon.icns                 # macOS icon (generated from icon.png)
│   ├── dmg-background.png        # DMG installer background
│   └── entitlements.plist        # macOS app permissions
├── resources/
│   └── python-bundle/            # Created by: npm run bundle-python
│       └── venv/                 # Python virtual environment
├── electron/
│   ├── main.js                   # Python process manager & IPC
│   └── preload.js                # Electron preload/IPC bridge
├── python/
│   ├── ml_server.py              # Python FastAPI server
│   ├── requirements.txt           # Python dependencies
│   ├── api_manager.py
│   └── tests/                    # Python tests
├── src/
│   ├── app/
│   │   ├── layout.jsx
│   │   └── page.jsx              # Root Next.js page
│   ├── components/               # React components
│   └── services/                 # API clients
├── package.json                  # npm scripts and dependencies
├── next.config.js
├── BUILD_MACOS.md                # This file
├── MACOS_BUILD_QUICKSTART.md     # Quick reference (deprecated, see this file)
└── BUILD_INSTRUCTIONS.txt        # Summary (deprecated, see this file)
```

---

## Troubleshooting

### "Python not found" Error

**Problem:** App crashes with Python not found at runtime

**Solution:** Verify bundle was created:
```bash
ls -la resources/python-bundle/venv/bin/python
```

If missing, rebuild:
```bash
rm -rf resources/python-bundle
npm run bundle-python
```

### App Crashes on Startup

**Problem:** Python subprocess fails to start

**Debugging:**
```bash
# Run with verbose logging
npm run electron:debug

# Check the console output for errors
```

Check that `python/ml_server.py` is valid Python:
```bash
# Test Python script directly
resources/python-bundle/venv/bin/python python/ml_server.py
```

### "Cannot Open Because Developer Cannot Be Verified"

**Problem:** macOS blocks app from unknown developer (expected for unsigned apps)

**Solution:**
```bash
# Allow the app to run
xattr -d com.apple.quarantine "/Applications/pibu_ai.app"

# OR: Right-click the app → Open → Open
```

### Bundle Size Too Large (>500MB)

**Problem:** Bundled Python environment is too large

**Solution Options:**

1. **Download models on first run** instead of bundling them
2. **Use quantized model versions** (smaller file size)
3. **Remove unused dependencies** from `python/requirements.txt`

Check size:
```bash
du -sh resources/python-bundle/
```

### "Module not found" When Running App

**Problem:** Python can't import required packages at runtime

**Solution:**
```bash
# Rebuild Python bundle
rm -rf resources/python-bundle
npm run bundle-python

# Verify all dependencies are installed
resources/python-bundle/venv/bin/pip list
```

### npm Scripts Not Found

**Problem:** `npm run bundle-python` not recognized

**Solution:**
```bash
# Make sure you're in the right directory
pwd  # Should end with: .../src/frontend-react

# Check package.json has the scripts
grep "bundle-python" package.json

# Make script executable
chmod +x scripts/bundle-python-macos.sh
```

### "Port 3000 Already in Use"

**Problem:** Next.js dev server can't start on port 3000

**Solution:**
```bash
# Kill existing process
lsof -ti:3000 | xargs kill -9

# Or use a different port
PORT=3001 npm run dev
```

---

## Code Signing (Optional, Required for Distribution)

### Why Sign?

- Required for distribution via Mac App Store
- Prevents "Unknown Developer" warning for users
- Enables notarization for Gatekeeper acceptance

### Setup

1. Get an Apple Developer Certificate from [developer.apple.com](https://developer.apple.com)

2. Set environment variables:
   ```bash
   export APPLE_ID="your-email@icloud.com"
   export APPLE_ID_PASSWORD="app-specific-password"  # NOT your actual password!
   export APPLE_TEAM_ID="XXXXXXXXXX"                  # Get from Apple Developer
   ```

3. Update `forge.config.js` (in root directory):
   ```javascript
   osxSign: {
     identity: "Apple Development",  // or "Apple Distribution"
     'hardened-runtime': true,
     entitlements: './src/frontend-react/build-resources/entitlements.plist'
   }
   ```

4. Rebuild:
   ```bash
   npm run make-dmg
   ```

---

## Next Steps

### Apple Silicon (M1/M2/M3) Support

To build a universal binary supporting both Intel and Apple Silicon:

1. Update `forge.config.js` (in root):
   ```javascript
   packagerConfig: {
     arch: ['x64', 'arm64'],  // Build for both
   }
   ```

2. Rebuild:
   ```bash
   npm run make-dmg
   ```

This creates a universal app that works on both Intel and Apple Silicon Macs.

### Windows Support

To support Windows:

1. Create `build-resources/windows-icon.ico` (Windows app icon)

2. Update `forge.config.js` (in root) to add Windows maker:
   ```javascript
   makers: [
     // ... existing macOS maker ...
     {
       name: '@electron-forge/maker-squirrel',
       config: { iconUrl: 'https://your-server.com/icon.ico' }
     }
   ]
   ```

3. On a Windows machine, run:
   ```bash
   npm install
   npm run bundle-python
   npm run build
   npm run make
   ```

### Automated CI/CD

For GitHub Actions, see `.github/workflows/ci.yml` to add frontend build steps.

### Performance Optimization

- **Model Streaming**: Download ML models on first run instead of bundling
- **Code Splitting**: Split React bundles for faster loading
- **Native Modules**: Consider using native Python extensions for performance-critical code

---

## Support & References

**Documentation:**
- `../../../ELECTRON_BUILD_SETUP.md` - Technical overview of entire setup
- `electron/main.js` - Python process management and IPC communication
- `scripts/bundle-python-macos.sh` - Python bundling implementation

**Common Issues:**
1. Python detection logic: See `electron/main.js` → `resolvePythonBin()`
2. Bundling issues: See `scripts/bundle-python-macos.sh`
3. App logs: `~/Library/Application Support/pibu_ai/logs/`

**Tools:**
- [Electron Forge](https://www.electronforge.io/) - App packaging
- [electron-builder](https://www.electron.build/) - Alternative builder (not used, but reference)
- [PyInstaller](https://pyinstaller.org/) - Python packaging (not used here)

---

## Architecture Overview

### On-Device (Electron App)

- Next.js React UI (running in Electron's Chromium)
- Python FastAPI server (subprocess)
- Local data storage (encrypted, user's home directory)
- All personal data stays on device

### Cloud (Optional Future)

- LLM inference (Modal or cloud API)
- Model serving (not bundled)
- Stateless processing

See `../../../docs/architecture.pdf` for full system design.

---

**Last Updated:** December 2025
**Maintained by:** AC215 Group 127
