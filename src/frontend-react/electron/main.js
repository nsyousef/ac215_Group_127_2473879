/**
 * Electron main process.
 * Handles IPC requests from renderer and manages file-based data operations.
 */

const { app, BrowserWindow, ipcMain, shell } = require('electron');
const path = require('path');
const url = require('url');
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
// Old Electron file handlers removed - all data now managed by Python
// ============================================================================

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

const activeStreams = new Map(); // streamId -> { caseId, question }


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
      APP_DATA_DIR: app.getPath('userData'),
      PYTHONPATH: [pyCwd, process.env.PYTHONPATH || ''].filter(Boolean).join(path.delimiter),
    },
  });

  const state = {
    child,
    buffer: '',
    // id -> { resolve, reject, onChunk? }
    pending: new Map(),
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
        const { id } = msg;
        if (!id) {
          console.warn('Python message without id:', msg);
          continue;
        }

        const pendingReq = state.pending.get(id);
        if (!pendingReq) {
          // Could be a late chunk for an already-completed request; ignore
          continue;
        }

        const { resolve, reject, onChunk } = pendingReq;

        // If Python sends streaming chunks, we expect a field like `chunk`
        if (typeof onChunk === 'function' && Object.prototype.hasOwnProperty.call(msg, 'chunk')) {
          try {
            onChunk(msg.chunk, msg);
          } catch (err) {
            console.error('onChunk handler error:', err);
          }
        }

        // Decide if this message is FINAL or just a partial one.
        // A simple convention:
        //   - msg.done === true  -> final
        //   - OR msg.ok === true/false with result/error -> final
        const isFinal =
          msg.done === true ||
          Object.prototype.hasOwnProperty.call(msg, 'ok') ||
          Object.prototype.hasOwnProperty.call(msg, 'error');

        if (isFinal) {
          state.pending.delete(id);

          if (msg.ok === false || msg.error) {
            reject(new Error(msg.error || 'Python error'));
          } else {
            // Prefer msg.result if present, else whole msg
            resolve(
              Object.prototype.hasOwnProperty.call(msg, 'result') ? msg.result : msg
            );
          }
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

/**
 * pyRequest
 * @param {string} caseId
 * @param {string} cmd
 * @param {Object} data
 * @param {(chunk: any, msg?: any) => void} [onChunk] optional streaming callback
 */
function pyRequest(caseId, cmd, data, onChunk) {
  const state = spawnPythonForCase(caseId);
  const id = `${Date.now()}_${reqSeq++}`;

  // Only add case_id if not already present (for static methods that pass their own case_id)
  const payload = {
    id,
    cmd,
    data: data.case_id ? data : { ...data, case_id: caseId },
  };

  state.lastUsed = Date.now();

  return new Promise((resolve, reject) => {
    state.pending.set(id, { resolve, reject, onChunk });

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
ipcMain.handle('ml:getInitialPrediction', async (event, { caseId, imagePath, textDescription, userTimestamp }) => {
  if (!caseId) throw new Error('caseId required');
  if (!imagePath) throw new Error('imagePath required');
  return await pyRequest(caseId, 'predict', {
    image_path: imagePath,
    text_description: textDescription,
    user_timestamp: userTimestamp
  }, (chunk) => {
    // Emit chunk event to renderer
    event.sender.send('ml:streamChunk', { chunk });
  });
});

const activeInitialStreams = new Map();

ipcMain.on(
  'ml:getInitialPredictionStream:start',
  async (event, { streamId, caseId, imagePath, textDescription, userTimestamp }) => {
    if (!caseId) {
      event.sender.send('ml:getInitialPredictionStream:chunk', {
        streamId,
        done: true,
        error: 'caseId required',
      });
      return;
    }

    if (!imagePath) {
      event.sender.send('ml:getInitialPredictionStream:chunk', {
        streamId,
        done: true,
        error: 'imagePath required',
      });
      return;
    }

    const sender = event.sender;

    // Track active stream (optional, but matches what you do for chat)
    activeStreams.set(streamId, { caseId, imagePath, textDescription });

    try {
      const finalResponse = await pyRequest(
        caseId,
        'predict',
        {
          image_path: imagePath,
          text_description: textDescription,
          user_timestamp: userTimestamp,
        },
        (chunk) => {
          // stream each chunk out as it arrives
          sender.send('ml:getInitialPredictionStream:chunk', {
            streamId,
            done: false,
            chunk,          // string or partial object, frontend normalizes
          });
        }
      );

      // final message
      sender.send('ml:getInitialPredictionStream:chunk', {
        streamId,
        done: true,
        finalResponse,     // this is what your generator in mlClient sees at the end
      });
    } catch (err) {
      sender.send('ml:getInitialPredictionStream:chunk', {
        streamId,
        done: true,
        error: err?.message || 'Unknown error in initial prediction stream',
      });
    } finally {
      activeStreams.delete(streamId);
    }
  }
);

ipcMain.on('ml:getInitialPredictionStream:cancel', (event, { streamId }) => {
  // For now we just drop tracking; you could also kill the Python process
  if (activeStreams.has(streamId)) {
    activeStreams.delete(streamId);
  }
});


ipcMain.handle('ml:chatMessage', async (event, { caseId, question, userTimestamp }) => {
  if (!caseId) throw new Error('caseId required');
  if (!question) throw new Error('question required');
  return await pyRequest(caseId, 'chat', {
    question,
    user_timestamp: userTimestamp
  }, (chunk) => {
    // Emit chunk event to renderer
    event.sender.send('ml:streamChunk', { chunk });
  });
});

ipcMain.on('ml:chatMessageStream:start', async (event, { streamId, caseId, question, userTimestamp }) => {
  if (!caseId) {
    event.sender.send('ml:chatMessageStream:chunk', {
      streamId,
      done: true,
      error: 'caseId required',
    });
    return;
  }

  if (!question) {
    event.sender.send('ml:chatMessageStream:chunk', {
      streamId,
      done: true,
      error: 'question required',
    });
    return;
  }

  const sender = event.sender;

  // If you add real cancellation later, you can store some handle here.
  activeStreams.set(streamId, { caseId, question });

  try {
    const finalResponse = await pyRequest(
      caseId,
      'chat',
      {
        question,
        user_timestamp: userTimestamp,
      },
      (chunk) => {
        // Stream each chunk as it arrives
        sender.send('ml:chatMessageStream:chunk', {
          streamId,
          done: false,
          chunk, // can be string or object; frontend normalizes it
        });
      }
    );

    // Indicate completion to the renderer
    sender.send('ml:chatMessageStream:chunk', {
      streamId,
      done: true,
      // optional: pass finalResponse if you ever want it
      finalResponse,
    });
  } catch (err) {
    sender.send('ml:chatMessageStream:chunk', {
      streamId,
      done: true,
      error: err?.message || 'Unknown error in chat stream',
    });
  } finally {
    activeStreams.delete(streamId);
  }
});

ipcMain.handle('ml:saveBodyLocation', async (event, { caseId, bodyLocation }) => {
  if (!caseId) throw new Error('caseId required');
  if (!bodyLocation) throw new Error('bodyLocation required');
  // Static method but needs real case_id - pass case_id in data so pyRequest doesn't override
  return await pyRequest('static', 'save_body_location', {
    case_id: caseId,
    body_location: bodyLocation
  });
});

ipcMain.handle('ml:loadConversationHistory', async (event, { caseId }) => {
  if (!caseId) throw new Error('caseId required');
  return await pyRequest(caseId, 'load_conversation_history', {});
});

ipcMain.handle('data:saveDemographics', async (event, demographics) => {
  if (!demographics) throw new Error('demographics required');
  return await pyRequest('static', 'save_demographics', { demographics });
});

ipcMain.handle('data:loadDemographics', async () => {
  return await pyRequest('static', 'load_demographics', {});
});

ipcMain.handle('data:resetPythonData', async () => {
  return await pyRequest('static', 'reset_all_data', {});
});

ipcMain.handle('data:loadDiseases', async () => {
  const result = await pyRequest('static', 'load_diseases', {});
  // Python returns {diseases: [...]}, extract the array
  return result.diseases || [];
});

ipcMain.handle('data:saveDiseases', async (event, diseases) => {
  if (!diseases) throw new Error('diseases required');
  return await pyRequest('static', 'save_diseases', { diseases });
});

ipcMain.handle('data:loadCaseHistory', async (event, caseId) => {
  if (!caseId) throw new Error('caseId required');
  return await pyRequest('static', 'load_case_history', { case_id: caseId });
});

ipcMain.handle('data:saveCaseHistory', async (event, caseId, caseHistory) => {
  if (!caseId) throw new Error('caseId required');
  if (!caseHistory) throw new Error('caseHistory required');
  return await pyRequest('static', 'save_case_history', { case_id: caseId, case_history: caseHistory });
});

ipcMain.handle('data:addTimelineEntry', async (event, caseId, imagePath, note, date) => {
  if (!caseId) throw new Error('caseId required');
  if (!imagePath) throw new Error('imagePath required');
  if (!date) throw new Error('date required');
  return await pyRequest('static', 'add_timeline_entry', {
    case_id: caseId,
    image_path: imagePath,
    note: note || '',
    date: date
  });
});

// Save uploaded image to temp directory
ipcMain.handle('save-uploaded-image', async (event, caseId, filename, buffer) => {
  try {
    const tempDir = path.join(app.getPath('temp'), 'pibu_uploads');
    await fs.mkdir(tempDir, { recursive: true });

    const imagePath = path.join(tempDir, `${caseId}_${filename}`);
    await fs.writeFile(imagePath, Buffer.from(buffer));

    // Verify file was written and exists
    try {
      await fs.access(imagePath);
      console.log(`Image saved successfully: ${imagePath}`);
    } catch (accessError) {
      console.error(`Image file not accessible after write: ${imagePath}`, accessError);
      throw new Error(`Failed to verify image file: ${imagePath}`);
    }

    return imagePath;
  } catch (e) {
    console.error('Error saving uploaded image:', e);
    throw e;
  }
});

// Read image file and convert to base64 data URL for renderer display
ipcMain.handle('read-image-as-data-url', async (event, imagePath) => {
  try {
    const imageBuffer = await fs.readFile(imagePath);
    const base64 = imageBuffer.toString('base64');
    // Detect image type from extension
    const ext = path.extname(imagePath).toLowerCase();
    const mimeType = ext === '.png' ? 'image/png' : ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' : 'image/png';
    return `data:${mimeType};base64,${base64}`;
  } catch (e) {
    console.error('Error reading image:', e);
    throw e;
  }
});

// ============================================================================
// Window Creation
// ============================================================================

function createWindow() {
  const icon = path.join(__dirname, '..', 'build-resources', 'icon.png');
  mainWindow = new BrowserWindow({
    icon: icon,
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

  // Always open DevTools in dev mode for debugging
  if (isDev) {
    mainWindow.webContents.openDevTools();
    // Also add keyboard shortcut to toggle DevTools (Cmd+Option+I on Mac, Ctrl+Shift+I on Windows/Linux)
    mainWindow.webContents.on('did-finish-load', () => {
      console.log('Electron window loaded. DevTools should be open.');
      console.log('If DevTools is not visible, press Cmd+Option+I (Mac) or Ctrl+Shift+I (Windows/Linux)');
    });
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
  // On macOS, BrowserWindow.icon is ignored; set Dock icon explicitly in dev
  if (process.platform === 'darwin') {
    try {
      const dockIconPath = path.join(__dirname, '..', 'build-resources', 'icon.png');
      if (fsSync.existsSync(dockIconPath) && app.dock && app.dock.setIcon) {
        app.dock.setIcon(dockIconPath);
      }
    } catch (e) {
      console.warn('Failed to set macOS Dock icon:', e);
    }
  }
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
