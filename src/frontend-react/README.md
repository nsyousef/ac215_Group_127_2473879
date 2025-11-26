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

## Using the App

1. **Create/Load Case**: On launch, create a new case or load an existing one
2. **Add Disease Entry**: Upload image and add description/location
3. **Get AI Analysis**: Python orchestrates ML predictions and LLM explanations
4. **Track Over Time**: View progression across multiple entries
5. **Chat with AI**: Ask follow-up questions about your condition

## Repository Structure

```
src/frontend-react/
├── electron/              # Electron main process
│   ├── main.js           # App lifecycle, IPC bridge
│   └── preload.js        # Secure context bridge
├── python/               # Python backend (API orchestration)
│   ├── api_manager.py    # Core ML/API orchestration
│   ├── ml_server.py      # Read in commands from JavaScript and call Python functions
│   ├── inference_local/  # Local vision encoder (EfficientNet)
│   ├── tests/            # Unit/integration/system tests
│   ├── Dockerfile        # Test container
│   └── docker-shell.sh   # Test runner
├── src/                  # React frontend (Next.js)
│   ├── app/             # Main page
│   ├── components/      # React components
│   ├── contexts/        # State management (disease, profile)
│   └── services/        # IPC communication layer
└── public/              # Static assets
```

## Functionality and API Manager

### How It Works

```
┌─────────────┐
│   React UI  │  ← User interacts with disease tracker
└──────┬──────┘
       │ IPC (stdin/stdout)
┌──────▼─────────┐
│    Electron    │  ← Spawns Python process per case
└──────┬─────────┘
       │ subprocess
┌──────▼─────────────────────────────┐
│   API Manager (Python)              │
│   1. Vision Encoder (local)         │  ← Generate image embeddings
│   2. Cloud ML (Google Cloud Run)    │  ← Disease predictions
│   3. Modal LLM (MedGemma)          │  ← Medical explanations
└──────────────────────────────────────┘
```

### API Manager (`python/api_manager.py`)

Core orchestration layer that:

**1. Manages ML Pipeline**
```python
APIManager(case_id: str, dummy: bool = False)
  ├─ predict_disease(image, text) → predictions
  ├─ get_llm_explanation(predictions, metadata) → explanation
  └─ ask_followup_question(question, context) → answer
```

**2. Integrates Three API Types**
- **Cloud ML** (`inference-cloud-*.run.app`): Text embedding + disease prediction
- **Modal LLM** (`*.modal.run`): Medical explanations via MedGemma-4B/27B
- **Local Vision** (`inference_local/`): EfficientNet-based image encoding

**3. Handles Data Persistence**
- Case history, conversation logs, demographics
- JSON storage in system app data directory

### Quick Start (Development)
```bash
cd src/frontend-react
npm run dev-electron  # Launches Next.js + Electron with Python backend
```

### Test Python Backend Only
```bash
cd python
python3 ml_server.py --case-id test_case --dummy
```

### Production Build
```bash
npm run build          # Build Next.js
npm run electron-pack  # Package for macOS → dist/Pibu-AI-*.dmg
```

## Configuration

### Environment Variables (Python)
```bash
# Cloud ML APIs
export BASE_URL="https://inference-cloud-469023639150.us-east4.run.app"

# Modal LLM APIs (optional override)
export LLM_EXPLAIN_URL="https://tanushkmr2001--dermatology-llm-27b-dermatologyllm-explain.modal.run"
export LLM_FOLLOWUP_URL="https://tanushkmr2001--dermatology-llm-27b-dermatologyllm-ask-followup.modal.run"
```

### Data Storage
- **macOS**: `~/Library/Application Support/pibu-ai/`
- **Linux**: `~/.config/pibu-ai/`
- **Windows**: `%APPDATA%/pibu-ai/`

Each case stored as JSON: `{case_id}_history.json`, `{case_id}_chat.json`, etc.
