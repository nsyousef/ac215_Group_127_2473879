/* eslint-disable no-console */
const { execSync } = require('node:child_process');
const { existsSync, mkdirSync, writeFileSync } = require('node:fs');
const path = require('node:path');

const root = path.resolve(process.cwd());
const pyDir = path.join(root, 'python');
const venvDir = path.join(pyDir, '.venv');
const pyBin = path.join(venvDir, 'bin', 'python3');
const pipBin = path.join(venvDir, 'bin', 'pip3');
const reqFile = path.join(pyDir, 'requirements.txt');
const metaFile = path.join(pyDir, '.python-bin-path.json');

function run(cmd) {
  execSync(cmd, { stdio: 'inherit' });
}

try {
  if (!existsSync(pyDir)) mkdirSync(pyDir, { recursive: true });

  if (!existsSync(pyBin)) {
    console.log('[python] Creating venv at', venvDir);
    run(`python3 -m venv "${venvDir}"`);
  } else {
    console.log('[python] Using existing venv at', venvDir);
  }

  console.log('[python] Upgrading pip');
  run(`"${pyBin}" -m pip install --upgrade pip`);

  if (existsSync(reqFile)) {
    console.log('[python] Installing requirements from', reqFile);
    run(`"${pipBin}" install -r "${reqFile}"`);
  } else {
    console.warn('[python] No requirements.txt found at', reqFile);
  }

  writeFileSync(metaFile, JSON.stringify({ python: pyBin }, null, 2), 'utf8');
  console.log('[python] Wrote python bin meta to', metaFile);
} catch (e) {
  console.error('[python] Failed to set up Python environment:', e && e.message ? e.message : e);
  process.exit(1);
}
