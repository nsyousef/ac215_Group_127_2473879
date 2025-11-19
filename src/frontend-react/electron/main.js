/**
 * Electron main process.
 * Handles IPC requests from renderer and manages file-based data operations.
 */

const { app, BrowserWindow, ipcMain, shell } = require('electron');
const path = require('path');
const fs = require('fs').promises;
const isDev = require('electron-is-dev');
const { spawn } = require('child_process');
const fsSync = require('fs');

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
// IPC Handlers for Profile
// ============================================================================

ipcMain.handle('load-profile', async () => {
  try {
    const dataDir = getDataDir();
    const profilePath = path.join(dataDir, 'profile.json');

    try {
      const data = await fs.readFile(profilePath, 'utf-8');
      return JSON.parse(data);
    } catch (e) {
      if (e.code === 'ENOENT') {
        return null; // No profile saved yet
      }
      throw e;
    }
  } catch (e) {
    console.error('Error loading profile:', e);
    throw e;
  }
});

ipcMain.handle('save-profile', async (event, profile) => {
  try {
    const dataDir = getDataDir();
    const profilePath = path.join(dataDir, 'profile.json');
    await fs.writeFile(profilePath, JSON.stringify(profile, null, 2), 'utf-8');
    return profile;
  } catch (e) {
    console.error('Error saving profile:', e);
    throw e;
  }
});

// Open external links in system browser
ipcMain.handle('open-external', async (event, url) => {
  try {
    await shell.openExternal(url);
    return true;
  } catch (e) {
    console.error('Error opening external URL:', url, e);
    return false;
  }
});

// Reset all app data by removing the data directory
ipcMain.handle('reset-app-data', async () => {
  try {
    const dataDir = getDataDir();
    await fs.rm(dataDir, { recursive: true, force: true });
    // Recreate data dir to ensure future writes succeed
    await fs.mkdir(dataDir, { recursive: true });
    return true;
  } catch (e) {
    console.error('Error resetting app data:', e);
    throw e;
  }
});

// ============================================================================
// ML: Python Process Manager (one process per caseId)
// ============================================================================

const PY_IDLE_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes

const pyProcesses = new Map(); // caseId -> { child, buffer, pending, lastUsed }

function resolvePythonBin() {
  // Dev path to repo-local venv
  const devVenvPy = path.join(__dirname, '..', 'python', '.venv', 'bin', 'python3');
  if (fsSync.existsSync(devVenvPy)) return devVenvPy;

  // Optional meta file written by setup script
  const metaPath = path.join(__dirname, '..', 'python', '.python-bin-path.json');
  try {
    if (fsSync.existsSync(metaPath)) {
      const { python } = JSON.parse(fsSync.readFileSync(metaPath, 'utf8'));
      if (python && fsSync.existsSync(python)) return python;
    }
  } catch {}

  // Fallback to env or system
  return process.env.PYTHON || 'python3';
}

function spawnPythonForCase(caseId) {
  const existing = pyProcesses.get(caseId);
  if (existing) {
    existing.lastUsed = Date.now();
    return existing;
  }

  const scriptPath = path.join(__dirname, '..', 'python', 'ml_server.py');
  const pyBin = resolvePythonBin();
  const pyCwd = path.join(__dirname, '..', 'python');
  const child = spawn(pyBin, [scriptPath], {
    cwd: pyCwd,
    stdio: ['pipe', 'pipe', 'pipe'],
    env: {
      ...process.env,
      PYTHONPATH: [pyCwd, process.env.PYTHONPATH || ''].filter(Boolean).join(path.delimiter),
    },
  });

  const state = {
    child,
    buffer: '',
    pending: new Map(), // id -> { resolve, reject }
    lastUsed: Date.now(),
  };

  child.stdout.on('data', (chunk) => {
    state.buffer += chunk.toString();
    let idx;
    while ((idx = state.buffer.indexOf('\n')) >= 0) {
      const line = state.buffer.slice(0, idx);
      state.buffer = state.buffer.slice(idx + 1);
      if (!line.trim()) continue;
      try {
        const msg = JSON.parse(line);
        const { id, ok, result, error } = msg;
        const p = state.pending.get(id);
        if (p) {
          state.pending.delete(id);
          if (ok) p.resolve(result);
          else p.reject(new Error(error || 'Python error'));
        }
      } catch (e) {
        console.error('Failed to parse Python response:', e, line);
      }
    }
  });

  child.stderr.on('data', (chunk) => {
    console.error(`[PY ${caseId}]`, chunk.toString());
  });

  child.on('exit', (code, signal) => {
    console.warn(`Python process for case ${caseId} exited: code=${code} signal=${signal}`);
    // Reject any pending requests
    for (const [, p] of state.pending) {
      p.reject(new Error('Python process exited'));
    }
    state.pending.clear();
    pyProcesses.delete(caseId);
  });

  pyProcesses.set(caseId, state);
  return state;
}

let reqSeq = 0;
function pyRequest(caseId, cmd, data) {
  const state = spawnPythonForCase(caseId);
  const id = `${Date.now()}_${reqSeq++}`;
  const payload = { id, cmd, data: { ...data, case_id: caseId } };
  state.lastUsed = Date.now();

  return new Promise((resolve, reject) => {
    state.pending.set(id, { resolve, reject });
    try {
      state.child.stdin.write(JSON.stringify(payload) + '\n');
    } catch (e) {
      state.pending.delete(id);
      reject(e);
    }
  });
}

// Cleanup timer for idle Python processes
setInterval(() => {
  const now = Date.now();
  for (const [caseId, state] of pyProcesses.entries()) {
    if (now - state.lastUsed > PY_IDLE_TIMEOUT_MS) {
      console.log(`Killing idle Python process for case ${caseId}`);
      try { state.child.kill(); } catch {}
      pyProcesses.delete(caseId);
    }
  }
}, 60 * 1000);

// IPC handlers to call Python
ipcMain.handle('ml:getInitialPrediction', async (event, { caseId, image, textDescription }) => {
  if (!caseId) throw new Error('caseId required');
  return await pyRequest(caseId, 'predict', { image, text_description: textDescription });
});

ipcMain.handle('ml:chatMessage', async (event, { caseId, question }) => {
  if (!caseId) throw new Error('caseId required');
  if (!question) throw new Error('question required');
  return await pyRequest(caseId, 'chat', { question });
});

// ============================================================================
// Window Creation
// ============================================================================

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  const startURL = isDev
    ? 'http://127.0.0.1:3000'
    : url.format({
        pathname: path.join(__dirname, '../out/index.html'),
        protocol: 'file:',
        slashes: true,
      });

  mainWindow.loadURL(startURL);

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
