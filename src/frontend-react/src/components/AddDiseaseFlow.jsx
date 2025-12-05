'use client';

import { useState, useEffect, useRef } from 'react';
import { isElectron } from '@/utils/config';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Grid,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Box,
  TextField,
  IconButton,
  LinearProgress,
  CircularProgress,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import FileAdapter from '@/services/adapters/fileAdapter';
import { BODY_PART_DEFAULTS } from '@/lib/constants';
import BodyMapPicker, { getBodyPartFromCoordinates } from '@/components/BodyMapPicker';
import { useDiseaseContext } from '@/contexts/DiseaseContext';
import mlClient from '@/services/mlClient';

// All valid body parts in order
const BODY_PARTS = [
  'ear',
  'hair',
  'face',
  'neck',
  'chest',
  'abdomen',
  'shoulders',
  'upper arm',
  'lower arm',
  'hands',
  'groin',
  'thighs',
  'lower legs',
  'foot',
];

// Format body part name for display (capitalize properly)
function formatBodyPartLabel(bodyPart) {
  return bodyPart
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export default function AddDiseaseFlow({ open, onClose, onSaved, canCancel = true, onboardingBack, onStartAnalysis }) {
  const theme = useTheme();
  const fullScreen = useMediaQuery(theme.breakpoints.down('md'));
  const { addDisease } = useDiseaseContext();

  const [step, setStep] = useState(0);
  const [bodyPart, setBodyPart] = useState('');
  const [mapPos, setMapPos] = useState(null); // { leftPct, topPct }
  const [mapError, setMapError] = useState(null); // Error message if invalid selection
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [note, setNote] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const firstChunkReceivedRef = useRef(false);
  const currentCaseIdRef = useRef(null);

  useEffect(() => {
    if (!open) reset();
  }, [open]);

  // Cleanup stream subscription on unmount or when dialog closes
  useEffect(() => {
    return () => {
      // Cleanup handled in analyzeImage function
    };
  }, [open]);

  function reset() {
    setStep(0);
    setBodyPart('');
    setFile(null);
    setPreview(null);
    setNote('');
    setAnalyzing(false);
    setError(null);
    firstChunkReceivedRef.current = false;
    currentCaseIdRef.current = null;
  }

  // File handling
  const onFileSelected = (f) => {
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(f);
    setFile(f);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) onFileSelected(f);
  };

  const handleMapChange = (coords) => {
    // coords: { leftPct, topPct }
    setMapPos(coords);

    // Use bounding box detection to infer body part from coordinates
    const detectedPart = getBodyPartFromCoordinates(coords.leftPct, coords.topPct);

    if (detectedPart === 'invalid') {
      setMapError('Please select a valid body area');
      setBodyPart('');
    } else {
      setMapError(null);
      setBodyPart(detectedPart);
    }
  };

  const analyzeImage = async () => {
    console.log('analyzeImage called', { file, bodyPart, note, mapPos });
    setAnalyzing(true);
    setError(null);

    try {
      // Ensure we have map coordinates: if user picked from list only, set default based on label
      let finalMapPos = mapPos;
      if (!finalMapPos && bodyPart && BODY_PART_DEFAULTS[bodyPart]) {
        const defaults = BODY_PART_DEFAULTS[bodyPart];
        finalMapPos = {
          leftPct: defaults.leftPct,
          topPct: defaults.topPct,
        };
        setMapPos(finalMapPos);
      }

      // Generate case ID for this analysis
      const timestamp = Date.now();
      const diseaseId = `${timestamp}`;       // Disease ID (no prefix, just timestamp)
      const caseId = `case_${timestamp}`;     // Folder name for Python storage

      // Save uploaded file to temp location for Python to access via IPC
      let imagePath;
      if (window.electronAPI && window.electronAPI.saveUploadedImage) {
        const buffer = await file.arrayBuffer();
        const uint8Array = new Uint8Array(buffer);
        imagePath = await window.electronAPI.saveUploadedImage(
          caseId,
          file.name,
          uint8Array,
        );
      } else {
        throw new Error('Image upload is only available in Electron runtime.');
      }

      // Build body location object with coordinates and NLP label
      const bodyLocation = {
        coordinates: finalMapPos ? [finalMapPos.leftPct, finalMapPos.topPct] : null,
        nlp: bodyPart || 'Unknown',
      };

      // Step 1: Save body location FIRST (before APIManager instantiation)
      await mlClient.saveBodyLocation(caseId, bodyLocation);

      // Step 2: Show analyzing step
      setStep(5);
      firstChunkReceivedRef.current = false;
      currentCaseIdRef.current = caseId;
      let receivedPredictionText = null;

      // Step 3: Subscribe to predictionText FIRST (before streaming)
      // Navigate immediately when predictionText arrives - this happens before LLM starts
      let predictionTextUnsubscribe = null;
      if (isElectron() && window.electronAPI?.mlOnPredictionText) {
        predictionTextUnsubscribe = window.electronAPI.mlOnPredictionText((predictionText) => {
          if (predictionText && currentCaseIdRef.current === caseId) {
            receivedPredictionText = predictionText;
            // Navigate immediately with predictionText and timeline data
            if (!firstChunkReceivedRef.current) {
              firstChunkReceivedRef.current = true;
              setAnalyzing(false);
              setStep(null);

              // Create timeline entry with initial image
              const currentDate = new Date().toISOString().split('T')[0];
              const timelineEntry = {
                id: 0,
                date: currentDate,
                image: imagePath, // Use the saved image path
                note: note || '',
              };

              const tempDisease = {
                id: diseaseId,
                name: 'Analyzing...', // Placeholder, will be updated
                image: preview,
                bodyPart: bodyPart || 'Unknown',
                mapPosition: finalMapPos ? {
                  leftPct: finalMapPos.leftPct,
                  topPct: finalMapPos.topPct
                } : null,
                description: '',
                confidenceLevel: 0,
                date: currentDate,
                llmResponse: '', // Will be populated later
                predictionText: predictionText, // Set predictionText immediately
                timelineData: [timelineEntry], // Include initial image in timeline
                conversationHistory: [],
                caseId: caseId,
              };

              // Navigate to results view immediately
              if (onStartAnalysis) {
                onStartAnalysis(tempDisease);
              }

              close();

              // Clean up predictionText subscription (don't unsubscribe from chunks - let them go to chat)
              if (predictionTextUnsubscribe) {
                predictionTextUnsubscribe();
              }
            }
          }
        });
      }

      // REMOVED: Chunk-based navigation - we navigate on predictionText instead
      // This prevents consuming the first chunk, allowing it to go to the chat panel

      // Step 4: Kick off prediction request (this will trigger LLM streaming on the backend)
      // Navigation should already have happened via predictionText, so we just wait for results
      const predictionPromise = mlClient.getInitialPrediction(
        imagePath,      // file path
        note || '',     // text description
        caseId,
        {},             // metadata
      );

      // Fallback: If predictionText doesn't arrive within 5 seconds, navigate anyway
      // This handles edge cases where predictionText might not be sent
      const fallbackTimeout = setTimeout(() => {
        if (!firstChunkReceivedRef.current && currentCaseIdRef.current === caseId) {
          console.warn('No predictionText received, navigating anyway after timeout');
          firstChunkReceivedRef.current = true;

          setAnalyzing(false);
          setStep(null);

          // Create timeline entry with initial image
          const currentDate = new Date().toISOString().split('T')[0];
          const timelineEntry = {
            id: 0,
            date: currentDate,
            image: imagePath,
            note: note || '',
          };

          const tempDisease = {
            id: diseaseId,
            name: 'Analyzing...',
            image: preview,
            bodyPart: bodyPart || 'Unknown',
            mapPosition: finalMapPos ? {
              leftPct: finalMapPos.leftPct,
              topPct: finalMapPos.topPct
            } : null,
            description: '',
            confidenceLevel: 0,
            date: currentDate,
            llmResponse: '',
            predictionText: '', // Will be updated when results arrive
            timelineData: [timelineEntry], // Include initial image
            conversationHistory: [],
            caseId: caseId,
          };

          if (onStartAnalysis) {
            onStartAnalysis(tempDisease);
          }

          close();

          if (predictionTextUnsubscribe) {
            predictionTextUnsubscribe();
          }
        }
      }, 5000); // 5 second timeout (predictionText should arrive much faster)

      // Wait for the full result in the background
      const results = await predictionPromise;

      // Clear fallback timeout if prediction completes
      clearTimeout(fallbackTimeout);

      // Find disease with highest confidence from predictions
      const predictions = results.predictions || {};
      let topDisease = 'Unknown Condition';
      let maxConfidence = 0;

      Object.entries(predictions).forEach(([disease, confidence]) => {
        if (confidence > maxConfidence) {
          maxConfidence = confidence;
          topDisease = disease;
        }
      });

      // Format disease name (capitalize first letter of each word)
      const formattedName = topDisease
        .split('_')
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');

      // Get the enriched disease object from Python response
      const enrichedDisease = results.enriched_disease || {};

      const newDisease = {
        id: diseaseId,
        name: formattedName,
        image: preview,
        description: enrichedDisease.description || '',
        bodyPart: enrichedDisease.bodyPart || bodyPart || 'Unknown',
        mapPosition: enrichedDisease.mapPosition || null,
        confidenceLevel: enrichedDisease.confidenceLevel || 0,
        date: enrichedDisease.date || '',
        llmResponse: enrichedDisease.llmResponse || '',
        predictionText: enrichedDisease.predictionText || '',
        timelineData: enrichedDisease.timelineData || [],
        conversationHistory: enrichedDisease.conversationHistory || [],
      };

      await addDisease(newDisease);

      // Update the selected condition with full data (it was already set to tempDisease)
      // The results view will re-render with the updated data
      if (onSaved) onSaved(newDisease);

      // Clean up predictionText subscription if still active
      if (predictionTextUnsubscribe) {
        predictionTextUnsubscribe();
      }

      // Note: onStartAnalysis was already called earlier with tempDisease (when predictionText arrived)
      // The selectedCondition will be updated when addDisease triggers a re-render
    } catch (e) {
      console.error('Analysis failed:', e);
      console.error('Error details:', {
        message: e.message,
        stack: e.stack,
        name: e.name,
      });

      // Clean up predictionText subscription on error
      if (predictionTextUnsubscribe) {
        predictionTextUnsubscribe();
      }
      firstChunkReceivedRef.current = false;
      currentCaseIdRef.current = null;

      const errorMessage = e.message || 'Analysis failed. Please try again.';
      setError(errorMessage);
      setAnalyzing(false);
      setStep(2); // Go back to photo step on error
    }
  };



  const handleStartAnalysis = async () => {
    console.log('handleStartAnalysis called', { file, bodyPart, step, hasFile: !!file, hasBodyPart: !!bodyPart });

    if (!file) {
      console.error('No file selected');
      setError('Please upload an image first.');
      setStep(2); // Go back to photo step
      return;
    }
    if (!bodyPart) {
      console.error('No body part selected');
      setError('Please select a body part first.');
      setStep(0); // Go back to body part selection
      return;
    }

    console.log('Starting analyzeImage...');
    try {
      // Start analysis and immediately close dialog to show chat
      await analyzeImage();
      console.log('analyzeImage completed successfully');
    } catch (error) {
      console.error('Error in analyzeImage:', error);
      console.error('Error stack:', error.stack);
      setError(`Failed to start analysis: ${error.message || error}`);
      setAnalyzing(false);
      setStep(2); // Go back to photo step on error
    }
  };

  const close = () => {
    reset();
    onClose && onClose();
  };

  const handleDialogClose = (event, reason) => {
    if (!canCancel && (reason === 'backdropClick' || reason === 'escapeKeyDown')) {
      return; // prevent closing during onboarding
    }
    if (analyzing || step === 5) {
      return; // prevent closing during analysis
    }
    close();
  };

  return (
    <Dialog open={open} fullScreen={fullScreen} onClose={handleDialogClose} disableEscapeKeyDown={!canCancel || step === 5} fullWidth maxWidth="sm">
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h6" component="div">Add Condition</Typography>
        {canCancel && step !== 5 && (
          <IconButton onClick={close} size="small" disabled={analyzing}><CloseIcon /></IconButton>
        )}
      </DialogTitle>

      <DialogContent dividers sx={{ overflow: 'auto', WebkitOverflowScrolling: 'touch' }}>
        {/* Step 0: Choose body part (map picker + list fallback) */}
        {step === 0 && (
          <Box>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>Select Body Position</Typography>
            <Typography variant="body2" sx={{ color: '#666', mb: 2 }}>Tap the body map to place the marker, or choose from the list below.</Typography>

            <Box sx={{ mb: 2 }}>
              <BodyMapPicker value={mapPos} onChange={(coords) => handleMapChange(coords)} />
            </Box>
            {mapError && (
              <Typography variant="body2" sx={{ color: 'error.main', mb: 2, textAlign: 'center' }}>
                {mapError}
              </Typography>
            )}

            <Typography variant="body2" sx={{ color: '#666', mb: 1 }}>Or pick from the list</Typography>
            <List>
              {BODY_PARTS.map((p) => (
                <ListItem key={p} disablePadding>
                  <ListItemButton
                    selected={bodyPart === p}
                    onClick={() => {
                      setBodyPart(p);
                      setMapError(null);
                      // Use default coordinates for this body part if available
                      if (BODY_PART_DEFAULTS[p]) {
                        const defaults = BODY_PART_DEFAULTS[p];
                        setMapPos({ leftPct: defaults.leftPct, topPct: defaults.topPct });
                      } else {
                        setMapPos(null);
                      }
                    }}
                  >
                    <ListItemText primary={formatBodyPartLabel(p)} />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          </Box>
        )}

        {/* Step 1: Coin / instructions */}
        {step === 1 && (
          <Box>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>Image Instructions</Typography>
            <Typography variant="body2" sx={{ color: '#666', mb: 2 }}>
              To enable tracking of size and color, place a coin (or a similarly sized object) near the skin condition in the photo.
            </Typography>
            <Box sx={{ my: 2, display: 'flex', justifyContent: 'center' }}>
              <img src="/assets/instructions.png" alt="Coin placement instructions" style={{ maxWidth: 320, width: '100%', borderRadius: 8, boxShadow: '0 2px 8px #0001' }} />
            </Box>
          </Box>
        )}

        {/* Step 2: Photo capture/upload */}
        {step === 2 && (
          <Box>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>Take or Upload Photo</Typography>
            <Typography variant="body2" sx={{ color: '#666', mb: 2 }}>
              Use your camera or upload an image. You can drag & drop or use the file picker.
            </Typography>

            <Box
              onDragOver={(e) => {
                // Only prevent default when files are being dragged to avoid interfering with touch scrolling
                try {
                  const types = e.dataTransfer && e.dataTransfer.types;
                  const hasFiles = types && (Array.from(types).includes('Files') || types.contains && types.contains('Files'));
                  if (hasFiles) e.preventDefault();
                } catch (err) {
                  // Fallback: do not prevent default to avoid blocking scroll
                }
              }}
              onDrop={handleDrop}
              sx={{ border: '2px dashed #ddd', borderRadius: 1, p: 2, textAlign: 'center' }}
            >
              {!preview ? (
                <>
                  <input
                    id="file-input"
                    type="file"
                    accept="image/*"
                    capture="environment"
                    style={{ display: 'none' }}
                    onChange={(e) => onFileSelected(e.target.files[0])}
                  />
                  <label htmlFor="file-input">
                    <Button component="span" variant="outlined">Choose file or take photo</Button>
                  </label>
                  <Typography variant="caption" sx={{ display: 'block', mt: 1 }}>Or drag & drop an image here</Typography>
                </>
              ) : (
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                    <img src={preview} alt="preview" style={{ maxWidth: '100%', maxHeight: 360, borderRadius: 8 }} />
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
                    <Button size="small" variant="text" onClick={() => { setFile(null); setPreview(null); }}>Clear image</Button>
                  </Box>
                </Box>
              )}
            </Box>
          </Box>
        )}


        {/* Step 4: Notes */}
        {step === 4 && (
          <Box>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>Optional Notes</Typography>
            <Typography variant="body2" sx={{ color: '#666', mb: 2 }}>Add any symptoms or context (optional, max 1000 characters).</Typography>
            <TextField value={note} onChange={(e) => setNote(e.target.value.slice(0, 1000))} placeholder="Describe symptoms, duration, etc." multiline minRows={4} fullWidth />
            {error && (
              <Typography variant="body2" sx={{ color: 'error.main', mt: 2 }}>
                {error}
              </Typography>
            )}
          </Box>
        )}

        {step === 5 && (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              py: 4,
              width: '100%',
            }}
          >
            <CircularProgress sx={{ mb: 2 }} />
            <Typography variant="h6">Analyzing…</Typography>
            <Typography variant="body2" sx={{ color: '#666', mt: 1 }}>
              Please wait while we analyze your image
            </Typography>
          </Box>
        )}

      </DialogContent>

      <DialogActions sx={{ px: 3, py: 2 }}>
        {/* Navigation buttons for each step */}
        {step === 0 && (
          <Box sx={{ display: 'flex', gap: 2, width: '100%', justifyContent: 'space-between' }}>
            <Box>
              {onboardingBack ? (
                <Button variant="text" onClick={onboardingBack}>BACK</Button>
              ) : (
                canCancel && <Button variant="text" onClick={close}>Cancel</Button>
              )}
            </Box>
            <Box>
              <Button variant="contained" onClick={() => setStep(1)} disabled={!bodyPart}>Next</Button>
            </Box>
          </Box>
        )}

        {step === 1 && (
          <Box sx={{ display: 'flex', gap: 2, width: '100%', justifyContent: 'space-between' }}>
            <Button variant="text" onClick={() => setStep(0)}>Back</Button>
            <Box>
              {canCancel && <Button variant="text" onClick={close}>Cancel</Button>}
              <Button variant="contained" onClick={() => setStep(2)} sx={{ ml: 1 }}>Next</Button>
            </Box>
          </Box>
        )}

        {step === 2 && (
          <Box sx={{ display: 'flex', gap: 2, width: '100%', justifyContent: 'space-between' }}>
            <Button variant="text" onClick={() => setStep(1)}>Back</Button>
            <Box>
              {canCancel && <Button variant="text" onClick={close}>Cancel</Button>}
              <Button variant="contained" onClick={() => setStep(4)} disabled={!preview} sx={{ ml: 1 }}>Continue</Button>
            </Box>
          </Box>
        )}

        {step === 4 && (
          <Box sx={{ display: 'flex', gap: 2, width: '100%', justifyContent: 'space-between' }}>
            <Button variant="text" onClick={() => setStep(2)}>Back</Button>
            <Box>
              {canCancel && <Button variant="text" onClick={close}>Cancel</Button>}
              <Button
                variant="contained"
                onClick={handleStartAnalysis}
                disabled={analyzing || !file}
                sx={{ ml: 1 }}
              >
                Analyze
              </Button>
            </Box>
          </Box>
        )}

        {/* Step 5: No buttons, just show analyzing (buttons hidden) */}
        {step === 5 && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
            {/* No buttons during analysis */}
          </Box>
        )}

      </DialogActions>
    </Dialog>
  );
}
