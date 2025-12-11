# Frontend (Electron + Next.js)

Desktop frontend for pibu.ai, built with Next.js and Electron. Uses a local Python process for ML orchestration via IPC.

## Prerequisites

- **Node.js** 18+ (20 LTS recommended)
- **Python** 3.10+ (3.11 recommended) available as `python3`
- **Xcode Command Line Tools** (for native packages): `xcode-select --install`

## Quick Start

### Development

```bash
cd src/frontend-react
npm install             # Creates python/.venv and installs dependencies
npm run dev-electron    # Launches Next.js + Electron with live reload
```

### Production Build (macOS)

```bash
cd src/frontend-react
npm install
npm run bundle-python   # Bundle Python environment
npm run build           # Build Next.js
npm run make-dmg        # Create DMG installer
```

**Output:** `dist/Pibu.dmg` (drag-to-install format)

## Architecture

```
React UI (Next.js) → Electron IPC → Python Process → ML APIs
```

- **React UI**: Disease tracking interface
- **Electron**: Desktop shell, spawns Python per case
- **Python Backend**: Orchestrates ML predictions and LLM explanations
  - Local vision encoder (EfficientNet)
  - Cloud ML APIs (disease predictions)
  - Modal LLM (MedGemma medical explanations)

## Repository Structure

```
src/frontend-react/
├── electron/             # Electron main process (IPC bridge)
│   ├── main.js           # App lifecycle, Python process manager
│   └── preload.js        # Secure context bridge
├── python/               # Python backend
│   ├── api_manager.py    # ML/API orchestration
│   ├── ml_server.py      # IPC command handler
│   ├── inference_local/  # Local vision encoder
│   └── tests/            # Unit/integration/system tests
├── src/                  # React frontend
│   ├── app/              # Next.js pages
│   ├── components/       # React components
│   ├── contexts/         # State management
│   └── services/         # IPC communication
├── build-resources/      # App icons, entitlements
└── scripts/              # Build scripts (bundle-python, create-dmg)
```

## Configuration

### Environment Variables (Python)

```bash
export BASE_URL="https://inference-cloud-469023639150.us-east4.run.app"
export LLM_EXPLAIN_URL="https://tanushkmr2001--dermatology-llm-27b-dermatologyllm-explain.modal.run"
export LLM_FOLLOWUP_URL="https://tanushkmr2001--dermatology-llm-27b-dermatologyllm-ask-followup.modal.run"
export LLM_TIME_TRACKING_URL="https://tanushkmr2001--dermatology-llm-27b-dermatologyllm-time-tracking.modal.run"
```

### Data Storage

- **macOS**: `~/Library/Application Support/pibu-ai/`

Case data stored as JSON: `{case_id}_history.json`, `{case_id}_chat.json`, etc.
