# ML Integration Flow Diagram

## Complete User Journey

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER STARTS APP                                │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │   HOME / DISEASE     │
                        │   LIST VIEW          │
                        │  (initially empty)   │
                        └──────────────────────┘
                                   │
                          Click "Add Disease"
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     ADD DISEASE FLOW (Modal)                            │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 1: Select Body Position                                          │
│    • Tap body map OR select from list                                  │
│    • Saves: bodyPart + mapPosition {leftPct, topPct}                   │
│                                                                         │
│  Step 2: Instructions                                                  │
│    • "Place coin next to skin condition"                               │
│                                                                         │
│  Step 3: Upload Image                                                  │
│    • Camera or file picker                                             │
│    • Saves: preview (base64 image)                                     │
│                                                                         │
│  Step 4: Add Notes (Optional)                                          │
│    • Text description of symptoms                                      │
│    • Max 1000 characters                                               │
│                                                                         │
│  Step 5: ANALYZE (Click button)                                        │
│    │                                                                    │
│    ▼                                                                    │
│  ┌────────────────────────────────────────────┐                       │
│  │  mlClient.getInitialPrediction()           │                       │
│  │    • image: base64 preview                 │                       │
│  │    • textDescription: user notes           │                       │
│  │    • caseId: "case_{timestamp}"            │                       │
│  └────────────────────────────────────────────┘                       │
│                     │                                                  │
│          (3 second simulated processing)                               │
│                     │                                                  │
│                     ▼                                                  │
│  ┌────────────────────────────────────────────┐                       │
│  │  ML RESULTS RETURNED                       │                       │
│  │  {                                         │                       │
│  │    predictions: {                          │                       │
│  │      "eczema": 0.78,                       │                       │
│  │      "contact_dermatitis": 0.15,           │                       │
│  │      "psoriasis": 0.04, ...                │                       │
│  │    },                                      │                       │
│  │    cv_analysis: { area, color, ... },     │                       │
│  │    llm_response: "Full explanation...",    │                       │
│  │    text_description: "User notes"          │                       │
│  │  }                                         │                       │
│  └────────────────────────────────────────────┘                       │
│                     │                                                  │
│                     ▼                                                  │
│  ┌────────────────────────────────────────────┐                       │
│  │  DISEASE OBJECT CREATED                    │                       │
│  │  {                                         │                       │
│  │    id: timestamp,                          │                       │
│  │    name: "Eczema",              ◄─ Highest confidence             │
│  │    description: "Based on..." ◄─ First 50 chars of LLM            │
│  │    image: base64,               ◄─ User's upload                  │
│  │    bodyPart: "Left Upper Arm",                                    │
│  │    mapPosition: {leftPct, topPct},                                │
│  │    caseId: "case_...",                                            │
│  │    predictions: {...},          ◄─ Full ML results                │
│  │    cvAnalysis: {...},                                             │
│  │    llmResponse: "Full text",                                      │
│  │    textDescription: "..."                                         │
│  │  }                                         │                       │
│  └────────────────────────────────────────────┘                       │
│                     │                                                  │
│                     ▼                                                  │
│  • Added to DiseaseContext (in-memory)                                │
│  • Saved to FileAdapter (persisted)                                   │
│  • Initial time entry saved (image + notes)                           │
│  • Conversation entry saved (LLM response)                            │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                      Modal closes, navigate to results
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          RESULTS VIEW                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │
│  │  RESULTS PANEL  │  │ TIME TRACKING   │  │   CHAT PANEL    │       │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤       │
│  │ [User's Image]  │  │ [Timeline]      │  │ [Initial LLM    │       │
│  │                 │  │                 │  │  Response]      │       │
│  │ Title: Eczema   │  │ ┌─────────────┐ │  │                 │       │
│  │                 │  │ │ 2025-11-16  │ │  │ "Based on      │       │
│  │ Full LLM text:  │  │ │ [Image]     │ │  │  your          │       │
│  │ "Based on your  │  │ │ Initial     │ │  │  description   │       │
│  │  description... │  │ │ upload      │ │  │  ..."          │       │
│  │  this appears   │  │ └─────────────┘ │  │                 │       │
│  │  to be..."      │  │                 │  │ [Input box]     │       │
│  │                 │  │                 │  │ [Send button]   │       │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                        User clicks "Back" / "Home"
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          HOME VIEW (Updated)                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌──────────────────┐                      │
│  │  DISEASE LIST       │    │   BODY MAP       │                      │
│  ├─────────────────────┤    ├──────────────────┤                      │
│  │ ┌─────────────────┐ │    │                  │                      │
│  │ │ [Thumbnail]     │ │    │   [Human Body]   │                      │
│  │ │ Eczema          │ │    │                  │                      │
│  │ │ "Based on..."   │ │    │      • ◄──────────── Red dot at         │
│  │ └─────────────────┘ │    │   (38%, 35%)     │    selected pos     │
│  │                     │    │                  │                      │
│  └─────────────────────┘    └──────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                        User clicks on disease
                                   │
                                   ▼
                          Back to RESULTS VIEW
                                   │
                        User clicks chat input
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          CHAT INTERACTION                               │
├─────────────────────────────────────────────────────────────────────────┤
│  User types: "What cream should I use?"                                │
│                     │                                                   │
│                     ▼                                                   │
│  ┌────────────────────────────────────────────┐                        │
│  │  mlClient.chatMessage()                    │                        │
│  │    • caseId: "case_123456789"              │                        │
│  │    • userQuery: "What cream..."            │                        │
│  └────────────────────────────────────────────┘                        │
│                     │                                                   │
│          (1.5 second simulated processing)                             │
│                     │                                                   │
│                     ▼                                                   │
│  ┌────────────────────────────────────────────┐                        │
│  │  LLM RESPONSE RETURNED                     │                        │
│  │  {                                         │                        │
│  │    answer: "That's a great question...",   │                        │
│  │    conversation_history: [...]             │                        │
│  │  }                                         │                        │
│  └────────────────────────────────────────────┘                        │
│                     │                                                   │
│                     ▼                                                   │
│  Chat panel shows:                                                     │
│  ┌─────────────────────────────────────┐                              │
│  │ [Initial LLM Response]              │                              │
│  │                                     │                              │
│  │ User: "What cream should I use?"    │                              │
│  │                                     │                              │
│  │ Pibu: "That's a great question...   │                              │
│  │       For managing skin conditions  │                              │
│  │       like this, it's important..." │                              │
│  └─────────────────────────────────────┘                              │
│                                                                         │
│  • Message saved to FileAdapter                                        │
│  • Conversation persisted locally                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Electron App)                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  React Components                                                │ │
│  │  • AddDiseaseFlow.jsx                                            │ │
│  │  • ResultsPanel.jsx                                              │ │
│  │  • ChatPanel.jsx                                                 │ │
│  │  • ConditionListView.jsx                                         │ │
│  │  • TimeTrackingPanel.jsx                                         │ │
│  │  • BodyMapView.jsx                                               │ │
│  └────────────────┬─────────────────────────────────────────────────┘ │
│                   │                                                    │
│                   ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  Context & Services                                              │ │
│  │  • DiseaseContext (state management)                             │ │
│  │  • mlClient.js (ML backend interface) ◄──── NEW!                │ │
│  │  • FileAdapter (local storage)                                   │ │
│  └────────────────┬─────────────────────────────────────────────────┘ │
│                   │                                                    │
└───────────────────┼────────────────────────────────────────────────────┘
                    │
                    │ (Future: IPC via Electron)
                    │ (Current: Simulated responses)
                    │
                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     PYTHON BACKEND (api_manager.py)                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  APIManager Class                                                │ │
│  │  • get_initial_prediction()                                      │ │
│  │  • chat_message()                                                │ │
│  │  • _run_local_ml_model() ◄── TODO: Vision model                 │ │
│  │  • _run_cv_analysis() ◄── TODO: CV features                     │ │
│  │  • _run_cloud_ml_model() ◄── TODO: Cloud API                    │ │
│  │  • _call_llm_explain()                                           │ │
│  │  • _call_llm_followup()                                          │ │
│  └────────────────┬─────────────────────────────────────────────────┘ │
│                   │                                                    │
└───────────────────┼────────────────────────────────────────────────────┘
                    │
                    ├─────────► LOCAL STORAGE
                    │           • history/{case_id}.json
                    │           • conversations/{case_id}.json
                    │
                    └─────────► CLOUD APIs (Future)
                                • Cloud ML Model (disease predictions)
                                • Modal LLM (explanations & chat)
```

## Privacy-First Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER'S DEVICE (Electron)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Raw Image ──────────┐                                          │
│  (NEVER leaves device)│                                         │
│                       │                                         │
│                       ▼                                         │
│  ┌──────────────────────────────────────────────┐              │
│  │  Local Vision Model (TODO)                   │              │
│  │  • CNN or ViT inference                      │              │
│  │  • Generates 512-dim embedding               │              │
│  └──────────────┬───────────────────────────────┘              │
│                 │                                               │
│                 │ Image Embedding (abstract features)           │
│                 │                                               │
└─────────────────┼───────────────────────────────────────────────┘
                  │
                  │ ONLY embedding sent to cloud
                  │ (not raw image!)
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CLOUD (Stateless)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Image Embedding ─────┐                                         │
│  Text Description ────┼────► Cloud ML Model                     │
│  User Metadata ───────┘      (Text encoder + Classifier)        │
│                                                                 │
│                              Returns: Disease Predictions       │
│                                                                 │
│  Predictions ─────────────► LLM (Modal)                         │
│  Context                    • MedGemma 4b/27b                   │
│                             • Generates explanation             │
│                                                                 │
│                              Returns: LLM Response              │
│                                                                 │
│  ❌ NO raw images stored                                        │
│  ❌ NO conversation history stored                              │
│  ❌ NO personal data persisted                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
src/frontend-react/
├── src/
│   ├── components/
│   │   ├── AddDiseaseFlow.jsx ◄───────── MODIFIED (ML integration)
│   │   ├── ResultsPanel.jsx ◄──────────── MODIFIED (show ML results)
│   │   ├── ConditionListView.jsx ◄────── MODIFIED (image thumbnails)
│   │   ├── ChatPanel.jsx ◄──────────────── MODIFIED (LLM integration)
│   │   ├── TimeTrackingPanel.jsx         (unchanged)
│   │   └── BodyMapView.jsx               (unchanged)
│   ├── contexts/
│   │   └── DiseaseContext.jsx            (unchanged)
│   ├── services/
│   │   ├── mlClient.js ◄────────────────── NEW! (ML backend interface)
│   │   └── diseaseService.js             (unchanged)
│   └── app/
│       └── page.jsx                       (unchanged)
├── python/
│   ├── api_manager.py ◄──────────────── MODIFIED (added helper methods)
│   ├── history/                          (created at runtime)
│   │   └── case_*.json
│   └── conversations/                    (created at runtime)
│       └── case_*.json
├── INTEGRATION_SUMMARY.md ◄────────────── NEW! (this document)
└── FLOW_DIAGRAM.md ◄───────────────────── NEW! (visual flows)
```
