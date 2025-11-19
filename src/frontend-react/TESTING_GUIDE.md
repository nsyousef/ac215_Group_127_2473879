# Testing Guide - ML Integration

## Quick Start Testing

### 1. Start the Development Server

```bash
cd src/frontend-react/
npm run dev
```

The app will open at `http://localhost:3000`

### 2. Test the Complete Flow

#### Step-by-Step Test Procedure:

**A. Initial State**
- [ ] App opens to home screen or onboarding
- [ ] Disease list is empty (or shows onboarding flow)
- [ ] Body map has no dots

**B. Add Disease Flow**
1. [ ] Click "Add Disease" button
2. [ ] Modal opens with "Select Body Position" step
3. [ ] Click on body map (e.g., left upper arm area)
4. [ ] Verify selected position shows on picker
5. [ ] Click "Next"
6. [ ] See coin instruction screen
7. [ ] Click "Next"
8. [ ] See photo upload screen
9. [ ] Upload an image (or use camera)
10. [ ] Verify image preview appears
11. [ ] Click "Continue"
12. [ ] See notes screen
13. [ ] Type some symptoms (e.g., "Red, itchy patch appeared 2 days ago")
14. [ ] Click "Analyze"

**C. Analysis Phase (3 seconds)**
- [ ] "Analyzing..." screen appears
- [ ] Progress bar shows
- [ ] Privacy message displays ("We never send your raw image...")

**D. Results View**
- [ ] Modal closes automatically
- [ ] Navigates to results view
- [ ] **Results Panel** shows:
  - [ ] Your uploaded image at the top
  - [ ] Disease name as title (e.g., "Eczema")
  - [ ] Full LLM explanation text (multiple paragraphs)
- [ ] **Time Tracking Panel** shows:
  - [ ] One entry with your uploaded image
  - [ ] Current date
  - [ ] Your notes text
- [ ] **Chat Panel** shows:
  - [ ] One message from "Pibu" (the assistant)
  - [ ] Initial LLM explanation text
  - [ ] Empty input box ready for questions

**E. Navigate Back Home**
1. [ ] Click "Back" or "Home" button
2. [ ] **Disease List** shows:
   - [ ] One entry
   - [ ] Small thumbnail of your uploaded image (instead of avatar)
   - [ ] Disease name (e.g., "Eczema")
   - [ ] Brief description (first 50 chars + "...")
3. [ ] **Body Map** shows:
   - [ ] One red dot at the position you selected
   - [ ] Dot is highlighted (slightly larger) if disease is selected

**F. Test Chat Interaction**
1. [ ] Click on the disease in the list (if not already selected)
2. [ ] Navigate back to results view (if on mobile)
3. [ ] Click on chat panel
4. [ ] Type a follow-up question: "What creams should I use?"
5. [ ] Click send button (or press Enter)
6. [ ] Verify:
   - [ ] Your message appears immediately (right-aligned, blue background)
   - [ ] Loading spinner shows in send button
   - [ ] After ~1.5 seconds, assistant response appears (left-aligned, gray background)
   - [ ] Response is relevant to your question
   - [ ] Input box is re-enabled

**G. Test Multiple Diseases (Optional)**
1. [ ] Go back to home
2. [ ] Click "Add Disease" again
3. [ ] Select a different body position
4. [ ] Upload a different image
5. [ ] Complete the flow
6. [ ] Verify:
   - [ ] Both diseases appear in list
   - [ ] Both dots appear on body map
   - [ ] Clicking each disease shows its respective data
   - [ ] Each has its own chat history
   - [ ] Each has its own time tracking

## Expected Results

### Disease Object Structure
When you inspect the disease in DiseaseContext, you should see:

```javascript
{
  id: 1731744000000,                // Your timestamp
  name: "Eczema",                   // Or another condition based on dummy ML
  description: "Based on your description and the image analysi...",
  bodyPart: "Left Upper Arm",
  mapPosition: {
    leftPct: 38,
    topPct: 35
  },
  image: "data:image/jpeg;base64,/9j/4AAQSkZJRg...",  // Your image
  createdAt: "2025-11-16T20:30:00.000Z",
  caseId: "case_1731744000000",
  predictions: {
    eczema: 0.78,
    contact_dermatitis: 0.15,
    psoriasis: 0.04,
    tinea_corporis: 0.02,
    seborrheic_dermatitis: 0.01
  },
  cvAnalysis: {
    area: 8.4,
    color_profile: {
      average_Lab: [67.2, 18.4, 9.3],
      redness_index: 0.34,
      texture_contrast: 0.12
    },
    boundary_irregularity: 0.23,
    symmetry_score: 0.78
  },
  llmResponse: "Based on your description \"Red, itchy patch...\" and the image analysis...",
  textDescription: "Red, itchy patch appeared 2 days ago"
}
```

### ML Client Responses

**Initial Prediction Response:**
```javascript
{
  llm_response: "Based on your description... [long text]",
  predictions: {
    eczema: 0.78,
    contact_dermatitis: 0.15,
    // ... more predictions
  },
  cv_analysis: {
    area: 8.4,
    color_profile: { ... },
    // ... more metrics
  },
  embedding: [0.123, 0.456, ...],  // 512 numbers
  text_description: "Your input text",
  case_id: "case_1731744000000"
}
```

**Chat Response:**
```javascript
{
  answer: "That's a great question about \"What creams should I use?\"...",
  conversation_history: []
}
```

## Visual Verification Checklist

### Home View
```
┌──────────────────────────────────────────┐
│ [Back]  Home                    [Profile]│
├──────────────────────────────────────────┤
│                                          │
│  Disease List         Body Map           │
│  ┌────────────┐      ┌──────────┐       │
│  │[Thumbnail] │      │          │       │
│  │ Eczema     │      │  [Body]  │       │
│  │ "Based..." │      │    •     │ ◄── Red dot
│  └────────────┘      └──────────┘       │
│                                          │
└──────────────────────────────────────────┘
```

### Results View (Desktop)
```
┌────────────────────────────────────────────────────────────┐
│ [Back]  Condition Detail                                   │
├────────────────────────────────────────────────────────────┤
│  Results Panel    │  Time Tracking   │  Chat Panel        │
│  ┌──────────────┐ │ ┌──────────────┐ │ ┌──────────────┐  │
│  │ [Your Image] │ │ │ 2025-11-16   │ │ │ Pibu: "Based │  │
│  │              │ │ │ [Image]      │ │ │  on your..."  │  │
│  │ Eczema       │ │ │ Initial      │ │ │              │  │
│  │              │ │ │ upload       │ │ │ [Input]      │  │
│  │ Based on...  │ │ └──────────────┘ │ │ [Send]       │  │
│  └──────────────┘ │                  │ └──────────────┘  │
└────────────────────────────────────────────────────────────┘
```

## Common Issues & Solutions

### Issue: Image not showing in results
**Check:**
- [ ] Image was successfully uploaded in step 3
- [ ] Preview appeared before clicking "Continue"
- [ ] Disease object has `image` property (check DevTools)

**Solution:** The image should be stored as base64 data URL. If not showing, check browser console for errors.

### Issue: Disease name is "Unknown Condition"
**Cause:** No predictions returned or all predictions are 0

**Expected:** Should be "Eczema" (highest confidence in dummy data)

**Check:** DevTools console for ML client response

### Issue: Chat not working
**Check:**
- [ ] Disease has `caseId` property
- [ ] Chat panel shows initial LLM response
- [ ] No errors in console when sending message

**Solution:** Verify `mlClient.chatMessage()` is being called with correct caseId

### Issue: Body map dot not appearing
**Check:**
- [ ] Disease has `mapPosition` property with `leftPct` and `topPct`
- [ ] Coordinates are numbers (not strings)
- [ ] Coordinates are reasonable (0-100 range)

**Solution:** If map position is null, dot won't show. Ensure body position was selected in flow.

### Issue: Time tracking empty
**Check:**
- [ ] Disease has `image` property
- [ ] FileAdapter is available (or using bundled data)

**Solution:** Initial time entry is created in `AddDiseaseFlow`. Check if `FileAdapter.saveTimeEntry()` succeeded.

## Browser DevTools Testing

### 1. Check Disease State
```javascript
// Open DevTools Console
// Type:
React.useState  // Find the DiseaseContext state
// Or use React DevTools extension to inspect DiseaseContext
```

### 2. Check ML Client Calls
```javascript
// In mlClient.js, add console.log at top of methods:
async getInitialPrediction(...) {
  console.log('ML Client: getInitialPrediction called', arguments);
  // ...
}
```

### 3. Monitor Network (Future)
When real Python integration is added:
- Network tab should show IPC calls
- No images should be sent over network (only embeddings)

## Performance Expectations

### Timing (Dummy Mode)
- Image upload: Instant
- Analysis phase: 3 seconds (simulated)
- Chat response: 1.5 seconds (simulated)

### Real ML (Future)
- Local vision model: ~2-5 seconds
- Cloud predictions: ~5-10 seconds
- LLM explanation: ~10-15 seconds
- Total: ~20-30 seconds

## Data Persistence Testing

### Browser Storage (Current)
- Diseases stored in memory (lost on refresh)
- FileAdapter attempts to persist but may fail in browser

### Electron Storage (Future)
- Diseases persisted to local filesystem
- Survives app restart
- History and conversations saved separately

## Mobile vs Desktop Testing

### Mobile Layout
- Bottom navigation on home/map views
- Back button on detail views
- Popover on body map tap
- Full-screen chat/time panels

### Desktop Layout
- 3-column home layout
- Side-by-side detail views
- No popover (direct selection)
- Persistent "Add Disease" button in list

## Test Data Examples

### Good Test Cases
1. **Simple condition:**
   - Body: "Face"
   - Image: Clear photo of skin
   - Notes: "Small red patch"

2. **Complex description:**
   - Body: "Left Upper Arm"
   - Image: Photo with coin for scale
   - Notes: "Red, itchy, spreading patch that appeared 3 days ago. Worse at night. No pain but uncomfortable."

3. **Minimal input:**
   - Body: "Torso"
   - Image: Any photo
   - Notes: "" (empty)

### Edge Cases to Test
- [ ] No notes (empty text)
- [ ] Very long notes (1000 chars)
- [ ] Large image files
- [ ] Multiple images uploaded sequentially
- [ ] Canceling mid-flow
- [ ] Selecting same body position twice

## Success Criteria

✅ **Integration Complete When:**
- [ ] Can add disease with image and notes
- [ ] Disease appears in list with image thumbnail
- [ ] Results show uploaded image and LLM text
- [ ] Time tracking shows initial entry
- [ ] Chat shows initial LLM response
- [ ] Can send follow-up questions
- [ ] Body map shows dot at selected position
- [ ] All data persists in disease object
- [ ] No console errors
- [ ] UI is responsive and smooth

## Next Testing Phase

When Python integration is added:
- [ ] Test real IPC communication
- [ ] Verify embedding generation (not raw images)
- [ ] Test cloud ML API calls
- [ ] Verify LLM endpoints
- [ ] Test error handling
- [ ] Test network failures
- [ ] Test large image processing
- [ ] Performance profiling
