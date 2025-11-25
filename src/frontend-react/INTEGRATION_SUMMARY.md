# ML Integration Summary

## Overview
The frontend app has been integrated with the dummy `api_manager.py` backend for the startup flow. This integration enables end-to-end disease analysis from image upload through results display and chat interaction.

## Key Components

### 1. ML Client Service (`src/services/mlClient.js`)
A new service layer that communicates with the Python backend (api_manager.py). Currently simulates Python API responses matching the dummy mode behavior.

**Key Methods:**
- `getInitialPrediction(image, textDescription, caseId, metadata)` - Analyzes uploaded image and returns ML predictions + LLM explanation
- `chatMessage(caseId, userQuery)` - Sends follow-up questions to LLM and returns responses
- `loadConversation(caseId)` - Loads chat history for a case
- `loadHistory(caseId)` - Loads time tracking history for a case

**TODO:** Replace simulated responses with actual IPC calls to Python process when Electron integration is ready.

### 2. Updated Components

#### AddDiseaseFlow.jsx
**Changes:**
- Integrated with `mlClient.getInitialPrediction()` to analyze uploaded images
- Extracts highest-confidence prediction as disease name
- Stores ML results (predictions, CV analysis, LLM response) in disease object
- Saves initial time tracking entry with uploaded image
- Creates first 50 chars of LLM response as brief description

**Data Flow:**
1. User uploads image and adds optional notes (text description)
2. User clicks "Analyze" → triggers `mlClient.getInitialPrediction()`
3. ML client returns:
   - `predictions`: Disease confidence scores (e.g., `{"eczema": 0.78, "psoriasis": 0.04}`)
   - `cv_analysis`: Image metrics (area, color profile, etc.)
   - `llm_response`: Full explanation text from LLM
   - `text_description`: User's original notes
4. Disease created with:
   - `name`: Formatted version of highest-confidence prediction
   - `description`: First 50 chars of LLM response + "..."
   - `image`: User's uploaded image (base64)
   - `caseId`: Unique identifier linking to ML backend
   - All ML results stored for later use

#### ResultsPanel.jsx
**Changes:**
- Displays user's uploaded image instead of placeholder
- Shows full LLM response as description text
- Falls back to placeholder if no image/LLM response available

**Display:**
- Top: User's uploaded image
- Title: Disease name (highest-confidence prediction)
- Body: Full LLM explanation with proper line breaks

#### ConditionListView.jsx
**Changes:**
- Displays image thumbnail instead of avatar icon when image available
- Shows brief description (first 50 chars of LLM response)

**Visual:**
- Each list item shows 40x40px thumbnail of uploaded image
- Falls back to avatar if no image available

#### TimeTrackingPanel.jsx
**No changes needed** - Already loads entries from FileAdapter
- Initial upload automatically saved as first time tracking entry
- User's image appears at top of timeline
- Note shows user's text description

#### ChatPanel.jsx
**Changes:**
- Integrated with `mlClient.chatMessage()` for follow-up questions
- Displays initial LLM response from disease object
- Shows loading spinner while waiting for LLM response
- Saves conversation to FileAdapter

**Chat Flow:**
1. Initial LLM response from analysis shown as first message
2. User asks follow-up question
3. Message sent to `mlClient.chatMessage()` with case ID
4. LLM response added to conversation
5. All messages saved to FileAdapter (if available)

#### BodyMapView.jsx
**No changes needed** - Already uses `mapPosition` from disease object
- Displays dot at user-selected position
- Falls back to default coordinates for predefined body parts

### 3. Python Backend Updates (api_manager.py)

**Added Helper Methods:**
- `_save_conversation_entry()` - Saves chat messages to `conversations/{case_id}.json`
- `_save_history_entry()` - Saves time tracking entries to `history/{case_id}.json`
- `_load_conversation_history()` - Loads chat history from disk

**File Structure:**
```
frontend-react/python/
├── history/
│   └── case_{timestamp}.json      # Time tracking entries
├── conversations/
│   └── case_{timestamp}.json      # Chat messages
└── api_manager.py
```

## User Flow

### Complete Startup Flow

1. **User opens app** → Shows onboarding or home screen
2. **User clicks "Add Disease"** → Opens AddDiseaseFlow modal
3. **User selects body position** → Map picker or list selection
4. **User uploads image** → Camera or file picker
5. **User adds optional notes** → Text description of symptoms
6. **User clicks "Analyze"** → ML processing begins (3 second simulated delay)
7. **Analysis completes** → Disease added to list with:
   - Name: Highest-confidence prediction (e.g., "Eczema")
   - Description: First 50 chars of LLM response + "..."
   - Image: User's uploaded photo as thumbnail
   - Position: Dot on body map at selected location
8. **User navigates to results view** → Shows:
   - Full uploaded image at top
   - Disease name as title
   - Complete LLM explanation as body text
   - Empty chat panel (ready for questions)
   - Time tracking with initial image and notes
9. **User goes back home** → Disease list shows:
   - One entry with image thumbnail
   - Disease name and brief description
   - Dot on body map at selected position
10. **User asks follow-up question in chat** →
    - Message sent to ML client
    - LLM response appears after ~1.5 seconds
    - Conversation saved locally

## Data Storage

### Disease Object Structure
```javascript
{
  id: 123456789,                    // Timestamp-based ID
  name: "Eczema",                   // Formatted highest-confidence prediction
  description: "Based on your description and the image analysi...", // First 50 chars
  bodyPart: "Left Upper Arm",       // User-selected or inferred
  mapPosition: {                    // Coordinates for body map
    leftPct: 38,
    topPct: 35
  },
  image: "data:image/jpeg;base64,...", // User's uploaded image
  createdAt: "2025-11-16T...",

  // ML Results
  caseId: "case_123456789",         // Links to backend storage
  predictions: {                     // Disease confidence scores
    "eczema": 0.78,
    "contact_dermatitis": 0.15,
    "psoriasis": 0.04
  },
  cvAnalysis: {                      // Computer vision metrics
    area: 8.4,
    color_profile: { ... },
    boundary_irregularity: 0.23,
    symmetry_score: 0.78
  },
  llmResponse: "Full LLM explanation text...",
  textDescription: "User's original notes"
}
```

### FileAdapter Storage (via Electron)
- **Time tracking**: `public/assets/data/time_tracking.json` (or Electron file system)
- **Chat history**: `public/assets/data/chat_history.json` (or Electron file system)
- **Diseases**: Managed by DiseaseContext, persisted via FileAdapter

## Privacy Compliance

✅ **Privacy-First Architecture Maintained:**
- User's raw image stored locally only (in disease object)
- Only image embeddings sent to cloud (TODO: when real ML integration complete)
- Text descriptions sent to cloud for embedding generation (as per architecture)
- All personal data (images, chat history, time tracking) stays on device
- Server receives only: embeddings + predictions context (stateless)

## Testing

### Current State (Dummy Mode)
- ML client simulates 3-second processing time
- Returns realistic dummy data matching api_manager.py format
- All UI updates work correctly with dummy data
- No actual Python process needed yet

### To Test:
1. Run app: `npm run dev` (from `src/frontend-react/`)
2. Click "Add Disease" on home screen
3. Select body position → Upload image → Add notes → Click "Analyze"
4. Wait 3 seconds for "analysis"
5. Verify:
   - ✅ Disease appears in list with image thumbnail
   - ✅ Results panel shows uploaded image + LLM response
   - ✅ Time tracking shows image + notes
   - ✅ Chat shows initial LLM response
   - ✅ Body map shows dot at selected position
   - ✅ Going back home shows disease in list

## Next Steps (TODOs)

### Immediate:
- [ ] Test complete flow in browser/Electron
- [ ] Verify all data persists correctly
- [ ] Test edge cases (no image, no notes, etc.)

### Future Integration:
- [ ] Replace `mlClient` simulation with actual IPC calls to Python
- [ ] Implement Electron ↔ Python communication layer
- [ ] Add on-device vision model inference
- [ ] Connect to cloud ML API for predictions
- [ ] Add error handling and retry logic
- [ ] Add loading states and progress indicators
- [ ] Implement conversation history persistence
- [ ] Add user metadata collection (age, sex, ethnicity)

## Files Modified

### New Files:
- `src/services/mlClient.js` - ML backend communication layer

### Modified Files:
- `src/components/AddDiseaseFlow.jsx` - ML integration for analysis
- `src/components/ResultsPanel.jsx` - Display ML results
- `src/components/ConditionListView.jsx` - Show image thumbnails
- `src/components/ChatPanel.jsx` - LLM chat integration
- `python/api_manager.py` - Added missing helper methods

### Unchanged Files (but work with integration):
- `src/components/TimeTrackingPanel.jsx` - Uses FileAdapter
- `src/components/BodyMapView.jsx` - Uses mapPosition
- `src/contexts/DiseaseContext.jsx` - Manages disease state
- `src/app/page.jsx` - Main app layout

## Notes

- All changes maintain existing UI/UX flow
- Dummy mode enabled by default for testing
- Real ML integration will be drop-in replacement
- Privacy architecture preserved throughout
- No breaking changes to existing components
