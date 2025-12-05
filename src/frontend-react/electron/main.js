/**
 * Electron main process.
 * Handles IPC requests from renderer and manages file-based data operations.
 */

// IMMEDIATE TEST: Write a marker file to prove this code runs
const fs_immediate = require('fs');
try {
  fs_immediate.writeFileSync('/tmp/pibu_main_js_loaded.txt', `Main.js loaded at ${new Date().toISOString()}\n`);
} catch (e) {}

console.log('🚀 pibu_ai Electron main process starting...');
console.error('🚀 pibu_ai Electron main process starting (stderr)...');
console.error('TEST_ERROR_OUTPUT_001');
process.stderr.write('STDERR_TEST_002\n');

// Import modules first
const { app, BrowserWindow, ipcMain, shell } = require('electron');
const path = require('path');
const url = require('url');
const fs = require('fs').promises;
const isDevModule = require('electron-is-dev');
// Handle both CommonJS and ES6 module exports
const isDev = typeof isDevModule === 'boolean' ? isDevModule : isDevModule.default || false;
const fs_module = require('fs');

// Write logs to file for debugging - write to /tmp for easier access
const logFile = '/tmp/pibu_ai_debug.log';
fs_module.writeFileSync(logFile, `\n========== PROCESS START ${new Date().toISOString()} ==========\n`, { flag: 'a' });

function debugLog(...args) {
  const msg = args.map(arg => typeof arg === 'string' ? arg : JSON.stringify(arg)).join(' ');
  const timestamp = new Date().toISOString();
  const fullMsg = `[${timestamp}] ${msg}\n`;
  console.log(fullMsg);
  console.error(fullMsg);  // Also log to stderr
  try {
    fs_module.appendFileSync(logFile, fullMsg, 'utf-8');
  } catch (e) {
    console.error('Failed to write to debug log:', e);
  }
}

debugLog('🚀 Main process loaded, modules imported, logFile at:', logFile);

// Global error handlers
process.on('uncaughtException', (error) => {
  console.error('💥 Uncaught exception:', error);
  if (logFile) debugLog('💥 Uncaught exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('💥 Unhandled rejection:', reason);
  if (logFile) debugLog('💥 Unhandled rejection at:', promise, 'reason:', reason);
  process.exit(1);
});
const { spawn } = require('child_process');
const fsSync = require('fs');
const http = require('http');

let mainWindow;
let productionServer = null;  // Local server for serving static files in production

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
  // Production: use bundled Python
  if (!isDev) {
    const bundledPython = path.join(
      process.resourcesPath,
      'python-bundle',
      'venv',
      'bin',
      'python'
    );
    if (fsSync.existsSync(bundledPython)) {
      console.log('✅ Using bundled Python:', bundledPython);
      return bundledPython;
    }
  }

  // Dev path to repo-local venv
  const devVenvPy = path.join(__dirname, '..', 'resources', 'python-bundle', 'venv', 'bin', 'python');
  if (fsSync.existsSync(devVenvPy)) {
    console.log('✅ Using dev venv Python:', devVenvPy);
    return devVenvPy;
  }

  // Fallback dev path
  const fallbackVenv = path.join(__dirname, '..', 'python', '.venv', 'bin', 'python3');
  if (fsSync.existsSync(fallbackVenv)) {
    console.log('✅ Using fallback venv Python:', fallbackVenv);
    return fallbackVenv;
  }

  // Optional meta file written by setup script
  const metaPath = path.join(__dirname, '..', 'python', '.python-bin-path.json');
  try {
    if (fsSync.existsSync(metaPath)) {
      const { python } = JSON.parse(fsSync.readFileSync(metaPath, 'utf8'));
      if (python && fsSync.existsSync(python)) {
        console.log('✅ Using Python from meta file:', python);
        return python;
      }
    }
  } catch {}

  // Fallback to system Python
  const systemPython = process.env.PYTHON || 'python3';
  console.log('⚠️  Falling back to system Python:', systemPython);
  return systemPython;
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

        const { resolve, reject, onChunk, eventSender } = pendingReq;

        // If Python sends predictionText metadata, forward it as a special event
        if (Object.prototype.hasOwnProperty.call(msg, 'predictionText')) {
          // Send predictionText as a special event before streaming starts
          if (eventSender) {
            eventSender.send('ml:predictionText', { 
              predictionText: msg.predictionText,
              reqId: id 
            });
          }
          return; // Don't treat this as final, continue waiting for chunks/result
        }

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
  }, event.sender);  // Pass event sender for predictionText forwarding
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
        },
        sender  // Pass event sender for predictionText forwarding
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

ipcMain.handle('data:deleteCases', async (event, caseIds) => {
  if (!caseIds || !Array.isArray(caseIds) || caseIds.length === 0) {
    throw new Error('caseIds must be a non-empty array');
  }
  return await pyRequest('static', 'delete_cases', { case_ids: caseIds });
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

// Simple static file server for production
function startProductionServer() {
  return new Promise((resolve, reject) => {
    const outDir = path.join(__dirname, '..', 'out');
    const publicDir = path.join(__dirname, '..', 'public');

    debugLog(`🔧 Setting up production server:`);
    debugLog(`   __dirname: ${__dirname}`);
    debugLog(`   outDir: ${outDir}`);
    debugLog(`   publicDir: ${publicDir}`);
    debugLog(`   outDir exists: ${fsSync.existsSync(outDir)}`);

    const server = http.createServer(async (req, res) => {
      try {
        debugLog(`📡 HTTP request: ${req.url}`);
        // Normalize the request path
        let filePath = req.url;
        if (filePath === '/') {
          filePath = '/index.html';
        }

        // Try to serve from out/ first (static exports), then public/
        let fullPath = path.join(outDir, filePath);

        let fileExists = false;
        try {
          await fs.stat(fullPath);
          fileExists = true;
        } catch (err) {
          // Try public directory
          fullPath = path.join(publicDir, filePath);
          try {
            await fs.stat(fullPath);
            fileExists = true;
          } catch (err2) {
            fileExists = false;
          }
        }

        if (!fileExists) {
          // File not found, serve index.html for client-side routing
          try {
            const indexPath = path.join(outDir, 'index.html');
            const content = await fs.readFile(indexPath);
            res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
            res.end(content);
            debugLog(`✅ Served index.html for route: ${req.url}`);
            return;
          } catch (err) {
            debugLog(`❌ Failed to read index.html:`, err);
            res.writeHead(404);
            res.end('Not found');
            return;
          }
        }

        // Read the file
        const content = await fs.readFile(fullPath);

        // Determine content type
        let contentType = 'text/plain';
        if (filePath.endsWith('.html')) contentType = 'text/html; charset=utf-8';
        else if (filePath.endsWith('.js')) contentType = 'application/javascript; charset=utf-8';
        else if (filePath.endsWith('.css')) contentType = 'text/css; charset=utf-8';
        else if (filePath.endsWith('.json')) contentType = 'application/json; charset=utf-8';
        else if (filePath.endsWith('.svg')) contentType = 'image/svg+xml';
        else if (filePath.endsWith('.png')) contentType = 'image/png';
        else if (filePath.endsWith('.jpg') || filePath.endsWith('.jpeg')) contentType = 'image/jpeg';
        else if (filePath.endsWith('.gif')) contentType = 'image/gif';
        else if (filePath.endsWith('.woff')) contentType = 'font/woff';
        else if (filePath.endsWith('.woff2')) contentType = 'font/woff2';

        res.writeHead(200, {
          'Content-Type': contentType,
          'Cache-Control': 'no-cache'
        });
        res.end(content);
        debugLog(`✅ Served file: ${filePath} (${content.length} bytes)`);
      } catch (err) {
        debugLog(`❌ Server error for ${req.url}:`, err);
        res.writeHead(500);
        res.end('Internal server error');
      }
    });

    server.listen(4000, '127.0.0.1', () => {
      debugLog('✅ Production static server running on http://127.0.0.1:4000');
      resolve(server);
    });

    server.on('error', (err) => {
      debugLog('❌ Server startup error:', err);
      reject(err);
    });
  });
}

function createWindow() {
  const icon = path.join(__dirname, '..', 'build-resources', 'icon.png');
  mainWindow = new BrowserWindow({
    icon: icon,
    width: 1200,
    height: 800,
    title: 'pibu.ai',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  let startURL;
  if (isDev) {
    startURL = 'http://127.0.0.1:3000';
    debugLog('🔧 Development mode - loading from:', startURL);
  } else {
    // In production, use local HTTP server to serve static files
    startURL = 'http://127.0.0.1:4000';
    debugLog('🔧 Production mode - loading from:', startURL);
  }

  debugLog(`📱 Loading URL: ${startURL}`);
  mainWindow.loadURL(startURL);

  // Never open dev tools automatically - users can use Cmd+Option+I if needed
  // if (isDev) {
  //   mainWindow.webContents.openDevTools();
  // }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
};

// ============================================================================
// App Lifecycle
// ============================================================================

app.on('ready', async () => {
  debugLog('📍 app.on(ready) event triggered');
  await ensureDataDirExists();
  debugLog('📍 Data directory ensured');

  // Start production server if not in dev mode
  debugLog('📍 isDev:', isDev);
  if (!isDev) {
    debugLog('📍 Production mode detected - starting HTTP server');
    try {
      productionServer = await startProductionServer();
      debugLog('✅ Production server started');
      // Give server a moment to be fully ready
      await new Promise(resolve => setTimeout(resolve, 100));
    } catch (error) {
      debugLog('❌ Failed to start production server:', error);
      // Still try to create the window and hope the server will start in time
    }
  } else {
    debugLog('📍 Development mode detected - skipping HTTP server');
  }

  // On macOS, BrowserWindow.icon is ignored; set Dock icon explicitly in dev
  if (process.platform === 'darwin') {
    try {
      const dockIconPath = path.join(__dirname, '..', 'build-resources', 'icon.png');
      if (fsSync.existsSync(dockIconPath) && app.dock && app.dock.setIcon) {
        app.dock.setIcon(dockIconPath);
      }
    } catch (e) {
      debugLog('Failed to set macOS Dock icon:', e);
    }
  }
  debugLog('📍 Creating window');
  createWindow();
});

app.on('window-all-closed', () => {
  // Clean up production server
  if (productionServer) {
    productionServer.close();
    productionServer = null;
  }

  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});
