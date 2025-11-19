# Dermatology LLM Assistant - Modal Deployment

Clean, minimal Modal deployment for the dermatology LLM assistant.

## 🚀 Quick Start

### Deploy with defaults (27b model on H200):
```bash
./deploy.sh
```

### Deploy 4b model:
```bash
./deploy.sh 4b
```

### Deploy 27b model on A100:
```bash
./deploy.sh 27b A100
```

### Deploy 4b model on specific GPU:
```bash
./deploy.sh 4b A100-80GB
```

## 📁 Structure

```
llm/
├── llm.py              # Core LLM logic
├── llm_modal.py        # Minimal Modal deployment
├── prompts.py          # System prompts
├── deploy.sh           # Simple deployment script
└── README.md           # This file
```

## ⚙️ Models

| Model | Default GPU | Max Tokens | Use Case |
|-------|-------------|------------|----------|
| `4b` | A100 | 500 | Faster, cheaper responses |
| `27b` | H200 | 700 | Better quality, more detailed |

## 🎮 GPU Options

Common GPU types:
- `A10G` - Cheapest, slower
- `A100` - Balanced
- `A100-80GB` - More memory
- `H200` - Fastest, most expensive

## 📡 API Endpoints

### `/explain` - Generate Initial Analysis
```python
import requests

response = requests.post(
    "https://your-url.modal.run/explain",
    json={
        "predictions": {"Eczema": 0.85, "Psoriasis": 0.10},
        "metadata": {
            "user_input": "Itchy red patches",
            "cv_analysis": {"area": 350, "color_profile": {"redness_index": 0.82}}
        }
    }
)
print(response.text)
```

### `/ask_followup` - Answer Follow-up Questions
```python
response = requests.post(
    "https://your-url.modal.run/ask_followup",
    json={
        "initial_answer": "...",
        "question": "What creams should I use?",
        "conversation_history": []
    }
)
result = response.json()
print(result['answer'])
```

## 🎯 Features

- ✅ **Thought Stopping** - Efficient token usage
- ✅ **Clean Responses** - No internal reasoning leaked
- ✅ **Conversation History** - Maintains last 5 questions
- ✅ **Dual Model Support** - 4b or 27b
- ✅ **Simple Deployment** - One command

## 🔧 Local Development

```python
from llm import LLM
from prompts import BASE_PROMPT, QUESTION_PROMPT

# Use 4b model
llm = LLM(
    model_name="medgemma-4b",
    max_new_tokens=500,
    base_prompt=BASE_PROMPT,
    question_prompt=QUESTION_PROMPT
)

# Or 27b model
llm = LLM(
    model_name="medgemma-27b",
    max_new_tokens=700,
    base_prompt=BASE_PROMPT,
    question_prompt=QUESTION_PROMPT
)

# Generate response
response = llm.explain(predictions={...}, metadata={...})
```

## 📊 Cost Optimization

**Development/Testing:**
```bash
./deploy.sh 4b A10G  # Cheapest option
```

**Production (Balanced):**
```bash
./deploy.sh 27b A100  # Good balance of cost/quality
```

**Production (Premium):**
```bash
./deploy.sh 27b H200  # Best quality
```

## 🛠️ Advanced Configuration

Edit `llm_modal.py` to change:
- `min_containers` - Minimum active containers (default: 0)
- `max_containers` - Maximum active containers (default: 1)
- `scaledown_window` - Scaledown delay in seconds (default: 1000)

## 📝 Notes

- First deployment: ~5-10 minutes (downloads model)
- Updates: ~1-2 minutes
- Requires Modal auth: `modal token new`
- Models cached in Modal Volume for fast restarts
