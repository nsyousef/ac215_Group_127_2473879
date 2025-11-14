# Electron Integration Guide

This guide explains how to integrate the Electron scaffolding with the Next.js frontend app.

## Architecture

The Electron setup provides:
- **Main Process** (`electron/main.js`): Handles window creation and IPC handlers for file operations
- **Preload Script** (`electron/preload.js`): Securely exposes IPC methods to the renderer process
- **FileAdapter** (`src/services/adapters/fileAdapter.js`): Communicates with Electron via `window.electronAPI`

### Data Flow

```
React Component (TimeTrackingPanel, ChatPanel)
        ↓
    FileAdapter (window.electronAPI.loadTimeTracking / loadChat)
        ↓
    Preload Script (electronAPI context bridge)
        ↓
    IPC Handler (ipcMain.handle('load-time-tracking', ...))
        ↓
    Main Process File I/O (fs.promises.readFile)
        ↓
    File System (userData/appData/{chat,time_tracking}/{conditionId}.json)
```

## Installation & Setup

### 1. Install Electron Dependencies

```bash
cd src/frontend-react
npm install electron electron-is-dev electron-builder concurrently wait-on
```

### 2. Merge Electron Package.json into Main package.json

Copy the build/scripts configuration from `electron-package.json` into the main `package.json`:
- Copy the `"main"` field to point to `electron/main.js`
- Copy the dev/build scripts that use `electron` and the build configurations
- Copy the new devDependencies

Or you can replace the entire `package.json` with the combined content.

### 3. Update Next.js Build Configuration

Edit `next.config.js` to export the app for Electron bundling:
```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export', // For Electron builds
  distDir: 'out',   // Build to 'out' folder
};

module.exports = nextConfig;
```

### 4. Update Electron Main Process

The `electron/main.js` assumes the following directory structure:
```
src/frontend-react/
  ├── electron/
  │   ├── main.js          (handles IPC)
  │   └── preload.js       (secure bridge)
  ├── out/                 (Next.js export output)
  ├── src/
  │   ├── app/
  │   ├── components/
  │   └── services/
  └── package.json
```

If your structure differs, update the paths in `electron/main.js`:
- `getDataDir()` — defaults to `app.getPath('userData') + '/appData'`
- `createWindow()` — loads from `http://localhost:3000` (dev) or `out/index.html` (prod)

## Running Electron

### Development (Hot Reload)

```bash
npm run dev-electron
```

This runs Next.js dev server on `http://localhost:3000` and Electron pointing to it. Changes to the Next.js app auto-reload in the Electron window.

### Production Build

```bash
npm run build-electron
```

This builds the Next.js app to `out/`, then uses electron-builder to package a standalone Electron app.

## File System Structure (App Data)

When running in Electron, the app stores data in the user's app data directory under `appData/`:

```
~/Library/Application Support/SkinCare App/appData/  (macOS)
%APPDATA%/SkinCare App/appData/                      (Windows)
~/.config/skincare-app/appData/                      (Linux)

appData/
  ├── diseases.json                 (optional; if provided, overrides bundled CONDITIONS)
  ├── chat/
  │   ├── 1.json                    (chat messages for condition id 1)
  │   ├── 2.json                    (chat messages for condition id 2)
  │   └── ...
  └── time_tracking/
      ├── 1.json                    (time entries for condition id 1)
      ├── 2.json                    (time entries for condition id 2)
      └── ...
```

## IPC Endpoints Reference

The following IPC methods are exposed via `window.electronAPI`:

### Diseases
- `loadDiseases()` → `Promise<Array | null>` — Loads diseases from `diseases.json` or null if file not found
- `getAppDataPath()` → `Promise<string>` — Returns the app data directory path

### Chat
- `loadChatHistory(conditionId)` → `Promise<Array>` — Loads chat messages for a condition
- `saveChatMessage(conditionId, message)` → `Promise<Object>` — Saves a new chat message

### Time Tracking
- `loadTimeTracking(conditionId)` → `Promise<Array>` — Loads time entries for a condition
- `saveTimeEntry(conditionId, entry)` → `Promise<Object>` — Saves a new time entry

## Example: Saving a Chat Message

```javascript
// In a React component or handler
const message = {
  id: 'msg123',
  conditionId: 2,
  role: 'user',
  text: 'Is this normal?',
  time: new Date().toISOString(),
};

try {
  const saved = await window.electronAPI.saveChatMessage(2, message);
  console.log('Message saved:', saved);
} catch (e) {
  console.error('Error saving message:', e);
}
```

## Next Steps

1. **Add "Add Entry" Flow** — Implement file upload and message sending to call `saveTimeEntry()` and `saveChatMessage()` IPC endpoints.
2. **Implement LLM Backend** — Replace placeholder chat responses with actual LLM integration (API or local model runner).
3. **Image Handling** — Implement image upload and storage for time tracking entries (save paths to app data or a media folder).

## Troubleshooting

### `window.electronAPI is undefined`

This means Electron is not running or the preload script failed to load. Check:
1. The app is running via `npm run dev-electron` or `electron .`
2. The `webPreferences.preload` path in `electron/main.js` is correct
3. No errors in the console (check Electron dev tools with `Ctrl+Shift+I` or `Cmd+Option+I`)

### Data not persisting

The app data directory must be writable. By default it's:
- macOS: `~/Library/Application Support/SkinCare App/appData/`
- Windows: `%APPDATA%\SkinCare App\appData\`
- Linux: `~/.config/skincare-app/appData/`

Ensure the app has write permissions.

### Next.js static export fails

If you see errors during `next build`, ensure `next.config.js` has `output: 'export'` enabled.
