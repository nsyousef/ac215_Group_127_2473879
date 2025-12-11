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

## What Remains Untested

The following files need more unit testing in this module:

<table class="index" data-sortable>
        <thead>
            <tr class="tablehead" title="Click to sort">
                <th id="file" class="name" aria-sort="none" data-shortcut="f">File<span class="arrows"></span></th>
                <th class="spacer">&nbsp;</th>
                <th id="statements" aria-sort="none" data-default-sort-order="descending" data-shortcut="s">statements<span class="arrows"></span></th>
                <th id="missing" aria-sort="none" data-default-sort-order="descending" data-shortcut="m">missing<span class="arrows"></span></th>
                <th id="excluded" aria-sort="none" data-default-sort-order="descending" data-shortcut="x">excluded<span class="arrows"></span></th>
                <th class="spacer">&nbsp;</th>
                <th id="coverage" aria-sort="none" data-shortcut="c">coverage<span class="arrows"></span></th>
            </tr>
        </thead>
        <tbody>
            <tr class="region">
                <td class="name"><a href="z_e1ac5446165bc0cf_api_manager_py.html">python&#8201;/&#8201;api_manager.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>767</td>
                <td>377</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="390 767">51%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_e1ac5446165bc0cf_config_loader_py.html">python&#8201;/&#8201;config_loader.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>13</td>
                <td>5</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="8 13">62%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_d5a87b97638658a9_module_py.html">python&#8201;/&#8201;cv-analysis&#8201;/&#8201;module.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>288</td>
                <td>287</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="1 288">1%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_d63344e941467bbb_transform_utils_py.html">python&#8201;/&#8201;inference_local&#8201;/&#8201;dataloader&#8201;/&#8201;transform_utils.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>7</td>
                <td>4</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="3 7">43%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_adfa1cdc8da53917_logger_py.html">python&#8201;/&#8201;inference_local&#8201;/&#8201;logger.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>12</td>
                <td>2</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="10 12">83%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_84fd819f5fc8e02c_cnn_py.html">python&#8201;/&#8201;inference_local&#8201;/&#8201;model&#8201;/&#8201;cnn.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>112</td>
                <td>96</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="16 112">14%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_adfa1cdc8da53917_vision_encoder_py.html">python&#8201;/&#8201;inference_local&#8201;/&#8201;vision_encoder.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>73</td>
                <td>49</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="24 73">33%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_e1ac5446165bc0cf_model_manager_py.html">python&#8201;/&#8201;model_manager.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>84</td>
                <td>35</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="49 84">58%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_e1ac5446165bc0cf_prediction_texts_py.html">python&#8201;/&#8201;prediction_texts.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>9</td>
                <td>1</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="8 9">89%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_b64e11b85e818c28_integration_py.html">python&#8201;/&#8201;tests&#8201;/&#8201;integration.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>63</td>
                <td>6</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="57 63">90%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_b64e11b85e818c28_system_py.html">python&#8201;/&#8201;tests&#8201;/&#8201;system.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>234</td>
                <td>62</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="172 234">74%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_b64e11b85e818c28_unit_py.html">python&#8201;/&#8201;tests&#8201;/&#8201;unit.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>224</td>
                <td>1</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="223 224">99%</td>
            </tr>
        </tbody>
    </table>
