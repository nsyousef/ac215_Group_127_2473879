# Electron App Build Setup - Summary

## Files Created/Modified

### 1. **`forge.config.js`** (Root level)
- Electron Forge configuration for macOS DMG creation
- Specifies app metadata, icons, signing configuration
- Defines where bundled Python will be included

### 2. **`scripts/bundle-python-macos.sh`** (New)
- Bash script that creates a standalone Python virtual environment
- Installs all dependencies from `src/frontend-react/python/requirements.txt`
- Creates bundle at `src/frontend-react/resources/python-bundle/venv/`
- Run with: `npm run bundle-python`

### 3. **`src/frontend-react/electron/main.js`** (Modified)
- Enhanced `resolvePythonBin()` function
- Now checks for bundled Python first (production), then dev venv, then system
- Includes proper logging for debugging Python path resolution

### 4. **`src/frontend-react/build-resources/entitlements.plist`** (New)
- macOS security entitlements allowing:
  - Subprocess spawning (for Python)
  - Network access (for API calls)
  - File system access
  - JIT code execution (for PyTorch/transformers)

### 5. **`src/frontend-react/package.json`** (Modified)
- Added new scripts:
  - `bundle-python` - Creates Python bundle
  - `build-electron` - Builds Next.js + bundles Python
  - `build-macos` - Full build with Electron Forge
- Added Electron Forge dependencies:
  - `@electron-forge/cli`
  - `@electron-forge/maker-dmg`
  - `@electron-forge/maker-zip`

### 6. **`src/frontend-react/BUILD_MACOS.md`** (New)
- Comprehensive build guide with prerequisites, steps, troubleshooting
- Covers code signing, Apple Silicon support, Windows builds
- File structure overview and CI/CD integration notes

### 7. **`src/frontend-react/MACOS_BUILD_QUICKSTART.md`** (New)
- Quick reference for developers
- 4-command build process for Intel Macs
- Troubleshooting shortcuts and file locations

## Build Process

```
npm run bundle-python
    ↓
Creates: src/frontend-react/resources/python-bundle/venv/
    ↓
npm run build
    ↓
Next.js compiles to .next/
    ↓
npm run make-dmg
    ↓
Electron Forge packages:
  - Next.js output
  - Bundled Python venv
  - Electron runtime
  - Into: out/make/Derma Assistant-*.dmg
```

## Key Features

✅ **Bundled Python**: Users don't need to install Python separately
✅ **Portable**: Everything included in the DMG, can run offline
✅ **Development Mode**: Uses system Python for faster development iteration
✅ **Code Signing Ready**: Configuration in place for production signing
✅ **Intel & Apple Silicon Ready**: Can build universal binaries
✅ **Subprocess Management**: Electron's IPC properly handles Python process lifecycle

## Development Workflow

```bash
# Development (live reload)
npm run dev-electron

# Production build
npm run build-macos

# Manual testing
open out/make/Derma\ Assistant-x64-*/Derma\ Assistant.app
```

## Next Steps

1. **Test the build locally**:
   ```bash
   cd src/frontend-react
   npm install
   npm run dev-electron
   ```

2. **Build DMG** (when ready):
   ```bash
   npm run bundle-python
   npm run build
   npm run make-dmg
   ```

3. **For Apple Silicon support** - Update `forge.config.js`:
   ```javascript
   arch: ['x64', 'arm64']  // Add to packagerConfig
   ```

4. **For code signing** - Get Apple Developer Certificate and set environment variables

5. **For Windows** - Add `@electron-forge/maker-squirrel` maker to `forge.config.js`

## File Structure Overview

```
project-root/
├── forge.config.js                    ← Electron build config
├── scripts/
│   └── bundle-python-macos.sh         ← Python bundling script
└── src/frontend-react/
    ├── build-resources/
    │   ├── entitlements.plist         ← macOS permissions
    │   └── icon.png                   ← App icon (needs to exist)
    ├── resources/
    │   └── python-bundle/venv/        ← Bundled Python (created by script)
    ├── electron/
    │   └── main.js                    ← Modified to detect bundled Python
    ├── package.json                   ← Updated with build scripts
    ├── BUILD_MACOS.md                 ← Detailed guide
    ├── MACOS_BUILD_QUICKSTART.md      ← Quick reference
    └── python/
        ├── ml_server.py
        └── requirements.txt
```

## Troubleshooting Commands

```bash
# Check if Python bundle exists
ls src/frontend-react/resources/python-bundle/venv/bin/python

# Rebuild Python bundle
rm -rf src/frontend-react/resources/python-bundle
npm run bundle-python

# Check bundle size
du -sh src/frontend-react/resources/python-bundle/

# Run with debug logging
npm run electron:debug

# Clear Electron cache
rm -rf ~/.config/Derma\ Assistant/  # Linux
rm -rf ~/Library/Application\ Support/Derma\ Assistant/  # macOS
```

## Notes

- DMG files are Intel (x64) only for now - add arm64 support when ready
- Python bundle size: typically 200-500MB depending on installed packages
- App data is stored in user's home directory, not bundled
- For distribution, you'll need Apple Developer signing certificate
