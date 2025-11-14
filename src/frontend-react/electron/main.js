/**
 * Electron main process.
 * Handles IPC requests from renderer and manages file-based data operations.
 */

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs').promises;
const isDev = require('electron-is-dev');

let mainWindow;

// Path to app data directory where diseases, chat, and time tracking data are stored
const getDataDir = () => {
  const dataDir = path.join(app.getPath('userData'), 'appData');
  return dataDir;
};

// Ensure data directory exists
const ensureDataDirExists = async () => {
  try {
    await fs.mkdir(getDataDir(), { recursive: true });
  } catch (e) {
    console.error('Error creating data directory:', e);
  }
};

// ============================================================================
// IPC Handlers for Diseases
// ============================================================================

ipcMain.handle('load-diseases', async () => {
  try {
    const dataDir = getDataDir();
    const diseasesPath = path.join(dataDir, 'diseases.json');

    // If file doesn't exist, return empty array (caller will use bundled CONDITIONS as fallback)
    try {
      const data = await fs.readFile(diseasesPath, 'utf-8');
      return JSON.parse(data);
    } catch (e) {
      if (e.code === 'ENOENT') {
        return null; // File not found, signal fallback to bundled data
      }
      throw e;
    }
  } catch (e) {
    console.error('Error loading diseases:', e);
    throw e;
  }
});

// Save diseases to disk
ipcMain.handle('save-diseases', async (event, diseases) => {
  try {
    const dataDir = getDataDir();
    const diseasesPath = path.join(dataDir, 'diseases.json');
    await fs.writeFile(diseasesPath, JSON.stringify(diseases, null, 2), 'utf-8');
    return diseases;
  } catch (e) {
    console.error('Error saving diseases:', e);
    throw e;
  }
});

// ============================================================================
// IPC Handlers for Chat History
// ============================================================================

ipcMain.handle('load-chat-history', async (event, conditionId) => {
  try {
    const dataDir = getDataDir();
    const chatDir = path.join(dataDir, 'chat');
    const chatPath = path.join(chatDir, `${conditionId}.json`);

    try {
      const data = await fs.readFile(chatPath, 'utf-8');
      return JSON.parse(data);
    } catch (e) {
      if (e.code === 'ENOENT') {
        return []; // No chat history for this condition yet
      }
      throw e;
    }
  } catch (e) {
    console.error(`Error loading chat history for condition ${conditionId}:`, e);
    throw e;
  }
});

ipcMain.handle('save-chat-message', async (event, conditionId, message) => {
  try {
    const dataDir = getDataDir();
    const chatDir = path.join(dataDir, 'chat');
    await fs.mkdir(chatDir, { recursive: true });

    const chatPath = path.join(chatDir, `${conditionId}.json`);
    let messages = [];

    // Load existing messages
    try {
      const data = await fs.readFile(chatPath, 'utf-8');
      messages = JSON.parse(data);
    } catch (e) {
      if (e.code !== 'ENOENT') throw e;
    }

    // Add new message
    messages.push(message);

    // Save back
    await fs.writeFile(chatPath, JSON.stringify(messages, null, 2), 'utf-8');
    return message;
  } catch (e) {
    console.error(`Error saving chat message for condition ${conditionId}:`, e);
    throw e;
  }
});

// ============================================================================
// IPC Handlers for Time Tracking
// ============================================================================

ipcMain.handle('load-time-tracking', async (event, conditionId) => {
  try {
    const dataDir = getDataDir();
    const timeDir = path.join(dataDir, 'time_tracking');
    const timePath = path.join(timeDir, `${conditionId}.json`);

    try {
      const data = await fs.readFile(timePath, 'utf-8');
      return JSON.parse(data);
    } catch (e) {
      if (e.code === 'ENOENT') {
        return []; // No time tracking entries for this condition yet
      }
      throw e;
    }
  } catch (e) {
    console.error(`Error loading time tracking for condition ${conditionId}:`, e);
    throw e;
  }
});

ipcMain.handle('save-time-entry', async (event, conditionId, entry) => {
  try {
    const dataDir = getDataDir();
    const timeDir = path.join(dataDir, 'time_tracking');
    await fs.mkdir(timeDir, { recursive: true });

    const timePath = path.join(timeDir, `${conditionId}.json`);
    let entries = [];

    // Load existing entries
    try {
      const data = await fs.readFile(timePath, 'utf-8');
      entries = JSON.parse(data);
    } catch (e) {
      if (e.code !== 'ENOENT') throw e;
    }

    // Add new entry
    entries.push(entry);

    // Save back
    await fs.writeFile(timePath, JSON.stringify(entries, null, 2), 'utf-8');
    return entry;
  } catch (e) {
    console.error(`Error saving time entry for condition ${conditionId}:`, e);
    throw e;
  }
});

// ============================================================================
// Utility IPC Handler
// ============================================================================

ipcMain.handle('get-app-data-path', () => {
  return getDataDir();
});

// ============================================================================
// Window Creation
// ============================================================================

const createWindow = () => {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  const startUrl = isDev
    ? 'http://localhost:3000' // Next.js dev server
    : `file://${path.join(__dirname, '../out/index.html')}`; // Built app

  mainWindow.loadURL(startUrl);

  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
};

// ============================================================================
// App Lifecycle
// ============================================================================

app.on('ready', async () => {
  await ensureDataDirExists();
  createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});
