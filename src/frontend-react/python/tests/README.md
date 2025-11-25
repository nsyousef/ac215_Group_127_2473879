# APIManager Tests

This directory contains comprehensive unit and integration tests for the APIManager class.

## Test Structure

- **`conftest.py`** - Shared pytest fixtures for both unit and integration tests
- **`unit.py`** - Unit tests with mocked dependencies (~23 tests)
- **`integration.py`** - Integration tests with real HTTP calls (~12 tests)

## Setup

### Install Dependencies

```bash
pip install -r ../requirements.txt
```

Or install testing dependencies separately:

```bash
pip install pytest pytest-mock
```

## Testing Philosophy

**Unit Tests:** All external dependencies are mocked to isolate logic.

**Integration Tests:**
- **TestCloudAPIs** makes REAL API calls once to verify endpoints work
- **All other tests** use MOCKED API responses for speed and reliability
- This approach gives us confidence that APIs work while keeping tests fast

**Benefits:**
- ✅ APIs verified once per test run (TestCloudAPIs)
- ✅ Other tests run 10x faster with mocked responses
- ✅ No repeated API costs for every test
- ✅ Tests remain reliable even if APIs are temporarily down

## Running Tests

### Unit Tests Only

Unit tests use mocked dependencies and do not require network connectivity:

```bash
cd /path/to/src/frontend-react/python
./run_unit_tests.sh
```

Or directly with pytest:

```bash
pytest tests/unit.py -v

# Run specific test class
pytest tests/unit.py::TestLoading -v
pytest tests/unit.py::TestSaving -v
pytest tests/unit.py::TestDeletion -v
pytest tests/unit.py::TestTextProcessing -v
pytest tests/unit.py::TestOther -v

# Run specific test
pytest tests/unit.py::TestLoading::test_init_creates_case_directory -v
```

### Integration Tests Only

Integration tests make **REAL HTTP calls** to cloud APIs and require:
- Network connectivity
- Valid API endpoints
- Modal CLI installed and configured (`pip install modal`)
- May take 2-5 minutes to complete

**Note:** The integration test script will:
1. Deploy the Modal LLM API automatically
2. Run the tests with real API calls
3. Clean up by stopping the Modal app

```bash
cd /path/to/src/frontend-react/python
./run_integration_tests.sh
```

Or directly with pytest:

```bash
pytest tests/integration.py -v -m integration

# Run specific test class
pytest tests/integration.py::TestCloudAPIs -v
pytest tests/integration.py::TestPredictionFlow -v
pytest tests/integration.py::TestChatFlow -v
pytest tests/integration.py::TestDiseaseLifecycle -v
pytest tests/integration.py::TestDataPersistence -v

# Run specific test
pytest tests/integration.py::TestCloudAPIs::test_cloud_text_embedding_api_returns_200 -v
```

### Run All Tests

```bash
pytest tests/ -v
```

## Test Coverage

The test suite targets **70-80% code coverage**, focusing on:

### Unit Tests (23 tests organized in 5 test classes)

**TestLoading (6 tests)**
- Initialization and directory creation
- JSON loading with fallbacks
- Demographics and diseases loading
- Enrichment from case history

**TestSaving (10 tests)**
- Demographics, body location, diseases
- Image saving with sequential naming
- Conversation and history entries
- Timeline entries and disease name updates

**TestDeletion (1 test)**
- Complete data reset functionality

**TestTextProcessing (3 tests)**
- Demographic augmentation (age, location)
- Fallback behavior with no demographics

**TestOther (3 tests)**
- Vision encoder initialization
- ML model embedding extraction
- Enriched disease object building

### Integration Tests (12 tests organized in 5 test classes)

**Testing Strategy:**
- **TestCloudAPIs**: Makes REAL API calls to verify endpoints work (slow, 1 time)
- **All other tests**: Use MOCKED API responses (fast, reliable)

**TestCloudAPIs (4 tests - REAL APIs)**
- Text embedding API (status only) ✓ Real API call
- Prediction API (status only) ✓ Real API call
- LLM explain API (status only) ✓ Real API call
- LLM followup API (status only) ✓ Real API call

**TestPredictionFlow (3 tests - Mocked APIs)**
- End-to-end workflow with mocked responses
- File creation verification
- Enriched disease object validation

**TestChatFlow (2 tests - Mocked APIs)**
- Mocked LLM followup responses
- Error handling without initial prediction

**TestDiseaseLifecycle (2 tests - No APIs)**
- Full CRUD workflow
- Multiple timeline entries

**TestDataPersistence (1 test - Mocked APIs)**
- Cross-instance data loading

## Functions Not Tested

The following functions are intentionally not tested to keep coverage at 70-80%:

**Utility Functions:**
- `debug_log()` - Simple stderr logging
- `_timestamp()` - Simple datetime wrapper

**Edge Case/Redundant Coverage:**
- `APIManager.load_case_history()` - Covered indirectly
- `APIManager.save_case_history()` - Covered indirectly
- `APIManager._run_cv_analysis()` - Marked as TODO, returns dummy data

**Low Priority:**
- `APIManager._load_conversation_history()` - Unused method
- `APIManager.get_vision_encoder()` lazy loading - Covered in integration

## Troubleshooting

### pytest not found
```bash
pip install pytest pytest-mock
```

### API endpoints unavailable
Integration tests will automatically skip if APIs are unreachable. Check:
- Network connectivity
- API endpoint URLs in `run_integration_tests.sh`
- Firewall settings

### Import errors
Make sure you're running tests from the `python/` directory:
```bash
cd src/frontend-react/python
pytest tests/unit.py
```

### Vision encoder checkpoint missing
Unit tests mock the vision encoder, so the checkpoint is not required. Integration tests also mock it to avoid loading the heavy model.

### Modal deployment fails
If Modal deployment fails during integration tests:
```bash
# Check Modal is installed and configured
modal --version
modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET

# Manually stop any running apps
modal app list
modal app stop dermatology-llm-4b
```

## Notes

- Unit tests run quickly (~5-10 seconds)
- Integration tests may take 2-5 minutes due to API calls
- Integration tests verify API status codes (200) but not response content
- All tests use temporary directories for file I/O (no cleanup needed)
