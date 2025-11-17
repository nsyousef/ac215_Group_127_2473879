# Frontend (Electron + Next.js)

This is the desktop frontend for Pibu.AI, built with Next.js (dev server) and Electron (desktop shell). It uses a local Python process for ML orchestration via IPC.

## Prerequisites (macOS)

- Node.js 18+ (20 LTS recommended) and npm
  - Check: `node -v`, `npm -v`
- Python 3.10+ (3.11 recommended) available as `python3`
  - Check: `python3 --version`
  - If missing: `brew install python@3.11`
- Xcode Command Line Tools (needed for some native npm packages)
  - Install: `xcode-select --install`

## Clone and Run (Dev)

```bash
# From anywhere
git clone https://github.com/nsyousef/ac215_Group_127_2473879.git
cd ac215_Group_127_2473879/src/frontend-react

# Install JS deps and set up a local Python venv with required packages
npm install  # runs a postinstall that creates python/.venv and installs python/requirements.txt

# Launch Next.js dev server + Electron
npm run dev-electron
```

What happens on first run:
- A Python virtual environment is created at `src/frontend-react/python/.venv/`
- Python deps (e.g., `requests`) are installed from `python/requirements.txt`
- Electron spawns a Python process per case (dummy mode), and the UI uses it via IPC

## Using the App (Dummy Mode)

1. Click “Add condition/disease”
2. Select a body position (map click or preset)
3. Upload an image and add optional notes
4. Click “Analyze”
   - The app calls the Python `APIManager(dummy=True)` via IPC
   - You’ll see predictions, an LLM explanation, and your uploaded image
5. Going back to Home shows the new condition with a thumbnail and a dot on the body map

## Data Storage (Dev)

- Python venv: `src/frontend-react/python/.venv/` (ignored by git)
- Case history: `src/frontend-react/python/history/`
- Chat conversations: `src/frontend-react/python/conversations/`
- These folders are local-only and should not be committed.

To clear local dev data:
```bash
rm -rf src/frontend-react/python/history src/frontend-react/python/conversations
```

## Troubleshooting

- Python not found or wrong version
  - Install Python: `brew install python@3.11`
  - Recreate venv: `npm run python:venv`
  - Force a specific Python: `PYTHON=/usr/local/bin/python3 npm run python:venv`
- Missing Python packages (e.g., ImportError: requests)
  - Re-run venv setup: `npm run python:venv`
- Electron doesn’t launch or crashes on start
  - Ensure CLT installed: `xcode-select --install`
  - Clean install: `rm -rf node_modules && npm install`
- Port already in use (Next.js default 3000)
  - Stop other dev servers or set a different port before running:
    ```bash
    PORT=3001 npm run dev-electron
    ```

## Notes

- Public dummy data under `public/assets/data/` is kept for reference but not loaded anymore.
- This README targets dev only. Packaging/bundling the Python env for production will be documented later.
