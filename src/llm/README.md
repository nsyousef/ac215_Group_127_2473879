# LLM Service

Modal-deployed dermatology assistant using MedGemma models for medical explanations and follow-up questions.

## Overview

Provides LLM-powered endpoints for:
- **Initial explanations**: Generate medical explanations from disease predictions
- **Follow-up questions**: Answer user questions with conversation context
- **Time tracking summaries**: Analyze progression of skin conditions over time
- **Streaming responses**: Real-time token streaming for better UX

## Models

- **MedGemma-4b**: Smaller, faster model (50 tokens)
- **MedGemma-27b**: Larger, more capable model (400-700 tokens)

## API Endpoints

- `POST /explain` - Generate explanation from predictions and metadata
- `POST /ask_followup` - Answer follow-up question with history
- `POST /ask_followup_stream` - Streaming follow-up (JSONL)
- `POST /explain_stream` - Streaming explanation (JSONL)
- `POST /time_tracking_summary` - Generate time progression summary

## Quick Start

### Deploy to Modal

```bash
# Manual deployment
export MODAL_MODEL_NAME="medgemma-27b"
export MODAL_MAX_TOKENS="700"
export MODAL_GPU="H200"
modal deploy llm_modal.py

# Or use helper script
./deploy.sh 27b H200
```

### Environment Variables

```bash
export MODAL_MODEL_NAME="medgemma-27b"  # or medgemma-4b
export MODAL_MAX_TOKENS="700"
export MODAL_GPU="H200"  # GPU type for Modal
```

## Architecture

- **llm.py**: Core LLM wrapper with MedGemma integration
- **llm_modal.py**: Modal deployment configuration
- **prompts.py**: Prompt templates for different use cases
- **Modal App**: Auto-scales with GPU support, caches model in volume

## Testing

```bash
# Unit tests
./run_unit_tests.sh

# Integration tests (requires Modal deployment)
./run_integration_tests.sh

# System tests (requires deployed endpoints)
./run_system_tests.sh
```

## What Remains Untested

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
            <td class="name"><a href="llm_py.html">llm.py</a></td>
            <td class="spacer">&nbsp;</td>
            <td>166</td>
            <td>101</td>
            <td>0</td>
            <td class="spacer">&nbsp;</td>
            <td data-ratio="65 166">39%</td>
        </tr>
        <tr class="region">
            <td class="name"><a href="llm_modal_py.html">llm_modal.py</a></td>
            <td class="spacer">&nbsp;</td>
            <td>53</td>
            <td>28</td>
            <td>1</td>
            <td class="spacer">&nbsp;</td>
            <td data-ratio="25 53">47%</td>
        </tr>
        <tr class="region">
            <td class="name"><a href="z_a44f0ac069e85531_system_py.html">tests&#8201;/&#8201;system.py</a></td>
            <td class="spacer">&nbsp;</td>
            <td>65</td>
            <td>34</td>
            <td>0</td>
            <td class="spacer">&nbsp;</td>
            <td data-ratio="31 65">48%</td>
        </tr>
        <tr class="region">
            <td class="name"><a href="z_a44f0ac069e85531_unit_py.html">tests&#8201;/&#8201;unit.py</a></td>
            <td class="spacer">&nbsp;</td>
            <td>71</td>
            <td>7</td>
            <td>0</td>
            <td class="spacer">&nbsp;</td>
            <td data-ratio="64 71">90%</td>
        </tr>
    </tbody>
</table>

## Deployment

Deployed via Pulumi (see `src/deployment/modules/modal_llm.py`) or manually using `deploy.sh`.

Service auto-scales (0-1 containers) with 1-hour scale-down window to minimize costs.
