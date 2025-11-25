# LLM Testing Suite

This directory contains unit tests and integration tests for the LLM module.

## Setup

Install test dependencies:

```bash
cd /Users/tk20/Desktop/APCOMP215/ac215_Group_127_2473879/src/llm
pip install -r requirements-test.txt
```

## Running Tests

### Run all tests

```bash
pytest
```

### Run only unit tests (excluding integration tests)

```bash
pytest -m "not integration"
```

### Run only fast tests (unit tests are fast with mocked models)

```bash
pytest tests/test_llm.py
```

### Run integration tests only

```bash
# First, set the Modal API URLs (separate endpoints for each function)
export MODAL_API_EXPLAIN_URL="https://your-org--dermatology-llm-4b-dermatologyllm-explain.modal.run"
export MODAL_API_ASK_FOLLOWUP_URL="https://your-org--dermatology-llm-4b-dermatologyllm-ask-followup.modal.run"

# Then run integration tests
pytest -m integration
```

### Run specific test file

```bash
pytest tests/test_llm.py
pytest tests/test_llm_modal_integration.py
```

### Run specific test class or function

```bash
pytest tests/test_llm.py::TestPromptBuilding
pytest tests/test_llm.py::TestPromptBuilding::test_build_prompt_with_all_components
```

### Run with verbose output

```bash
pytest -v
```

### Run with coverage

```bash
pytest --cov=llm --cov-report=html
```

## Test Structure

### Unit Tests (`test_llm.py`)

**Note**: Unit tests use mocked models to avoid loading large model weights. This makes tests fast and doesn't require GPU access.

1. **TestLLMModelLoading**: Tests model initialization
   - `test_model_loading`: Verifies LLM initializes with mocked components
   - `test_model_attributes`: Checks model attributes are set correctly

2. **TestPromptBuilding**: Tests prompt construction
   - `test_build_prompt_with_all_components`: Full metadata test
   - `test_build_prompt_without_history`: Without historical data
   - `test_build_prompt_minimal`: Minimal metadata

3. **TestGeneration**: Tests text generation
   - `test_generate_returns_text`: Basic generation works
   - `test_generate_filters_thoughts`: Removes "thought" markers
   - `test_generate_with_different_temperatures`: Temperature handling

4. **TestExplain**: Tests explanation generation
   - `test_explain_generates_response`: Produces substantive output
   - `test_explain_uses_predictions`: Uses prediction data
   - `test_explain_with_minimal_data`: Handles minimal input

5. **TestAskFollowup**: Tests follow-up questions
   - `test_ask_followup_returns_dict`: Correct return structure
   - `test_ask_followup_answer_content`: Non-empty answers
   - `test_ask_followup_updates_history`: History management
   - `test_ask_followup_limits_history`: History length limiting
   - `test_ask_followup_empty_history`: Handles no prior history

### Integration Tests (`test_llm_modal_integration.py`)

1. **TestExplainEndpoint**: Tests `/explain` API endpoint
   - `test_explain_successful_response`: Basic successful call
   - `test_explain_with_history`: With historical data
   - `test_explain_minimal_data`: Minimal input

2. **TestAskFollowupEndpoint**: Tests `/ask_followup` API endpoint
   - `test_ask_followup_successful_response`: Basic successful call
   - `test_ask_followup_with_history`: With conversation history
   - `test_ask_followup_no_history`: Without history
   - `test_ask_followup_multiple_questions`: Sequential questions

## Test Markers

- `@pytest.mark.integration`: Integration tests requiring API access

**Note**: Unit tests no longer use `slow` or `gpu` markers since they use mocked models.

## Environment Variables

- `MODAL_API_EXPLAIN_URL`: Required for integration tests. Set to your Modal /explain endpoint URL.
- `MODAL_API_ASK_FOLLOWUP_URL`: Required for integration tests. Set to your Modal /ask_followup endpoint URL.

Example:
```bash
export MODAL_API_EXPLAIN_URL="https://your-org--dermatology-llm-4b-dermatologyllm-explain.modal.run"
export MODAL_API_ASK_FOLLOWUP_URL="https://your-org--dermatology-llm-4b-dermatologyllm-ask-followup.modal.run"
```

## Test Data and Fixtures

### Mock Data (mock_data.py)
Static test data imported as global constants:
- `BASE_PROMPT`: Base prompt from prompts.py
- `QUESTION_PROMPT`: Question prompt from prompts.py
- `MOCK_PREDICTIONS`: Sample predictions dictionary
- `MOCK_METADATA`: Sample metadata with user input, CV analysis, and history
- `API_EXPLAIN_URL`: Modal API URL for /explain endpoint (from environment variable)
- `API_ASK_FOLLOWUP_URL`: Modal API URL for /ask_followup endpoint (from environment variable)

### Fixtures (conftest.py)
Only needed for creating mocked objects:
- `mock_llm`: LLM instance with mocked model and processor (fast, no GPU needed)
- `mock_model`: Mocked transformers model
- `mock_processor`: Mocked transformers processor

## Notes

- **Unit tests use mocked models** - they don't load real model weights, making them fast and not requiring GPU
- **Integration tests** test the actual deployed API and are automatically skipped if `MODAL_API_EXPLAIN_URL` or `MODAL_API_ASK_FOLLOWUP_URL` are not set
- Mocking is done using `pytest-mock` and Python's built-in `unittest.mock`
- All tests use pytest's built-in assertions for clarity
- Unit tests focus on testing the logic of prompt building, text filtering, and conversation history management
