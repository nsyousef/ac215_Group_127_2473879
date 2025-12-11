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
import { BODY_PART_DEFAULTS } from '@/lib/constants';
import BodyMapPicker from '@/components/BodyMapPicker';
import ImageCropper from '@/components/ImageCropper';
import { useDiseaseContext } from '@/contexts/DiseaseContext';
import mlClient from '@/services/mlClient';

// All valid body parts in order
const BODY_PARTS = [
  // Front
  'ear',
  'hair',
  'face',
  'neck',
  'chest',
  'abdomen',
  'shoulders',
  'armpit',
  'upper arm',
  'lower arm',
  'hands',
  'groin',
  'hips',
  'thighs',
  'lower legs',
  'foot',
  // Back
  'back of head',
  'upper back',
  'mid back',
  'lower back',
  'buttocks',
  'calves',
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
  const [croppedPreview, setCroppedPreview] = useState(null); // Cropped image data URL
  const [cropData, setCropData] = useState(null); // Crop coordinates { x, y, width, height }
  const [croppedFile, setCroppedFile] = useState(null); // Cropped file blob
  const [note, setNote] = useState('');
  const [hasCoin, setHasCoin] = useState(false); // Checkbox: image contains a coin (default: false)
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
    setCroppedPreview(null);
    setCropData(null);
    setCroppedFile(null);
    setNote('');
    setHasCoin(false); // Reset checkbox to default (unchecked)
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
    setError(null); // Clear any previous errors when new file is selected
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) onFileSelected(f);
  };

  const handleMapChange = (coords) => {
    // coords: { leftPct, topPct, bodyPart, isFrontView }
    setMapPos(coords);

    // Use body part from coords (already detected by BodyMapPicker)
    const detectedPart = coords?.bodyPart || 'invalid';

    if (detectedPart === 'invalid') {
      setMapError('Please select a valid body area');
      setBodyPart('');
    } else {
      setMapError(null);
      setBodyPart(detectedPart);
    }
  };

  const handleCropComplete = (croppedDataUrl, cropCoords, blob) => {
    console.log('Crop completed:', { cropCoords });
    setCroppedPreview(croppedDataUrl);
    setCropData(cropCoords);

    // Convert blob to File if available, otherwise use original file
    if (blob) {
      const croppedFile = new File([blob], file.name, { type: 'image/jpeg' });
      setCroppedFile(croppedFile);
    } else {
      setCroppedFile(file); // User skipped cropping
    }

    // Move to next step (notes)
    setStep(4);
  };

  const analyzeImage = async () => {
    console.log('analyzeImage called', { file, croppedFile, bodyPart, note, mapPos, cropData, hasCoin });
    console.log('[AddDiseaseFlow] hasCoin value:', hasCoin, 'type:', typeof hasCoin);
    setAnalyzing(true);
    setError(null);

    // Helper to clean Electron IPC error prefix
    const cleanErrorMessage = (msg) => {
      if (!msg) return '';
      // Remove "Error invoking remote method 'xxx': Error: " prefix
      return msg.replace(/^Error invoking remote method '[^']+': Error: /, '');
    };

    // Declare outside try block so error handler can access it
    let predictionTextUnsubscribe = null;

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

      // Use cropped file if available, otherwise original file
      const fileToUpload = croppedFile || file;
      const displayImage = croppedPreview || preview;

      // Save uploaded file to temp location for Python to access via IPC
      let imagePath;
      if (window.electronAPI && window.electronAPI.saveUploadedImage) {
        const buffer = await fileToUpload.arrayBuffer();
        const uint8Array = new Uint8Array(buffer);
        imagePath = await window.electronAPI.saveUploadedImage(
          caseId,
          fileToUpload.name || file.name,
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

      console.log('[AddDiseaseFlow] Body location saved, showing analyzing step');

      // Step 2: Show analyzing step with spinner
      setAnalyzing(true);
      setStep(5);
      console.log('[AddDiseaseFlow] Set step to 5, analyzing=true');

      firstChunkReceivedRef.current = false;
      currentCaseIdRef.current = caseId;

      // Step 3: Subscribe to predictionText event - this fires AFTER predictions but BEFORE streaming
      if (isElectron() && window.electronAPI?.mlOnPredictionText) {
        console.log('[AddDiseaseFlow] Subscribing to predictionText event for caseId:', caseId);
        predictionTextUnsubscribe = window.electronAPI.mlOnPredictionText((predictionText) => {
          console.log('[AddDiseaseFlow] predictionText event received:', {
            predictionText: predictionText?.substring(0, 100) + '...',
            currentCaseId: currentCaseIdRef.current,
            expectedCaseId: caseId,
            alreadyNavigated: firstChunkReceivedRef.current
          });
          if (predictionText && currentCaseIdRef.current === caseId && !firstChunkReceivedRef.current) {
            console.log('[AddDiseaseFlow] Conditions met, navigating to results now');
            firstChunkReceivedRef.current = true;

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
              name: 'Analyzing...', // Placeholder, will be updated
              image: displayImage,
              bodyPart: bodyPart || 'Unknown',
              mapPosition: finalMapPos ? {
                leftPct: finalMapPos.leftPct,
                topPct: finalMapPos.topPct
              } : null,
              description: '',
              confidenceLevel: 0,
              date: currentDate,
              llmResponse: '', // Will be populated by streaming
              predictionText: predictionText, // Set predictionText from Python
              timelineData: [timelineEntry],
              conversationHistory: [],
              caseId: caseId,
            };

            // Clean up subscription
            if (predictionTextUnsubscribe) {
              predictionTextUnsubscribe();
            }

            // Add disease to context BEFORE navigation so ChatPanel can find it
            console.log('[AddDiseaseFlow] Adding tempDisease to context');
            addDisease(tempDisease).then(() => {
              console.log('[AddDiseaseFlow] Disease added to context, now navigating');

              // Close the dialog immediately so navigation can take effect
              console.log('[AddDiseaseFlow] Closing dialog before navigation');
              setAnalyzing(false);
              onClose && onClose();

              // Navigate to results - streaming will start shortly after this
              if (onStartAnalysis) {
                console.log('[AddDiseaseFlow] Calling onStartAnalysis');
                onStartAnalysis(tempDisease);
                console.log('[AddDiseaseFlow] onStartAnalysis completed');
              }
            });
          }
        });
      }

      // Step 4: Set up fallback timeout in case predictionText doesn't arrive
      const fallbackTimeout = setTimeout(() => {
        if (!firstChunkReceivedRef.current && currentCaseIdRef.current === caseId) {
          console.warn('[AddDiseaseFlow] predictionText timeout, navigating anyway');
          firstChunkReceivedRef.current = true;

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
            image: displayImage,
            bodyPart: bodyPart || 'Unknown',
            mapPosition: finalMapPos ? {
              leftPct: finalMapPos.leftPct,
              topPct: finalMapPos.topPct
            } : null,
            description: '',
            confidenceLevel: 0,
            date: currentDate,
            llmResponse: '',
            predictionText: '',
            timelineData: [timelineEntry],
            conversationHistory: [],
            caseId: caseId,
          };

          if (predictionTextUnsubscribe) {
            predictionTextUnsubscribe();
          }

          // Add disease to context BEFORE navigation (fallback timeout case)
          addDisease(tempDisease).then(() => {
            // Close the dialog immediately so navigation can take effect
            setAnalyzing(false);
            onClose && onClose();

            if (onStartAnalysis) {
              onStartAnalysis(tempDisease);
            }
          });
        }
      }, 30000); // 30 second timeout (vision encoder + cloud API can be slow)

      // Step 5: Kick off prediction request and wait for initial validation
      // Check for UNCERTAIN before proceeding to navigation
      let results;
      try {
        results = await mlClient.getInitialPrediction(
          imagePath,      // file path
          note || '',     // text description
          caseId,
          { hasCoin },    // Pass hasCoin flag in metadata
        );

        // Check if this is an UNCERTAIN result (not an error, so no console logging)
        if (results && results.isUncertain) {
          // Clear fallback timeout
          clearTimeout(fallbackTimeout);

          // Clean up subscription
          if (predictionTextUnsubscribe) {
            predictionTextUnsubscribe();
          }

          const errorMsg = results.message || 'Unable to determine the condition from this image. Please try again with better lighting, a clearer photo, or more descriptive text. It\'s also possible that there\'s nothing concerning to identify.';
          setError(errorMsg);
          setAnalyzing(false);
          setStep(2); // Go back to image upload
          return;
        }
      } catch (e) {
        console.error('Prediction failed:', e);

        // Clear fallback timeout
        clearTimeout(fallbackTimeout);

        // Clean up subscription
        if (predictionTextUnsubscribe) {
          predictionTextUnsubscribe();
        }

        // Other errors (actual failures)
        const errorMsg = cleanErrorMessage(e.message) || 'Analysis failed. Please try again.';
        setError(errorMsg);
        setAnalyzing(false);
        setStep(2);
        return;
      }

      // Clear fallback timeout since prediction completed successfully
      clearTimeout(fallbackTimeout);

      // Clean up subscription if still active
      if (predictionTextUnsubscribe) {
        predictionTextUnsubscribe();
      }

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
        image: displayImage,
        description: enrichedDisease.description || '',
        bodyPart: enrichedDisease.bodyPart || bodyPart || 'Unknown',
        mapPosition: enrichedDisease.mapPosition || null,
        confidenceLevel: enrichedDisease.confidenceLevel || 0,
        date: enrichedDisease.date || '',
        llmResponse: enrichedDisease.llmResponse || '',
        predictionText: enrichedDisease.predictionText || '',
        timelineData: enrichedDisease.timelineData || [],
        conversationHistory: enrichedDisease.conversationHistory || [],
        caseId: caseId,
      };

      // Update the disease in context (it was already added as tempDisease)
      await addDisease(newDisease);

      // Notify parent that analysis is complete with full data
      if (onSaved) onSaved(newDisease);

      // Note: onStartAnalysis was already called earlier with tempDisease (via predictionText event)
      // The selectedCondition will be updated when addDisease triggers a re-render
    } catch (e) {
      console.error('Analysis failed:', e);
      console.error('Error details:', {
        message: e.message,
        stack: e.stack,
        name: e.name,
        code: e.code,
      });

      // Clean up subscription on error
      if (predictionTextUnsubscribe) {
        predictionTextUnsubscribe();
      }

      firstChunkReceivedRef.current = false;
      currentCaseIdRef.current = null;

      // Handle error (UNCERTAIN is handled above, so this is for actual failures)
      const errorMsg = cleanErrorMessage(e.message) || 'Analysis failed. Please try again.';
      setError(errorMsg);

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
              <BodyMapPicker
                value={mapPos}
                onChange={(coords) => handleMapChange(coords)}
                showBoundingBoxes={false}
              />
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

        {/* Step 1: Photo instructions */}
        {step === 1 && (
          <Box>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>Image Instructions</Typography>
            <Typography variant="body2" sx={{ color: '#666', mb: 2 }}>
              Please take a clear, well-focused photo of the skin area you want to analyze.
            </Typography>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" sx={{ color: '#666', mb: 1 }}>Make sure:</Typography>
              <ul style={{ marginTop: 0, paddingLeft: '1.2rem', color: '#666', fontSize: '0.9rem' }}>
                <li>The area is brightly and evenly lit</li>
                <li>There are no shadows, glare, or strong reflections</li>
                <li>The camera is held steady and the image is in focus</li>
                <li>Nothing is covering or touching the skin area (hair, clothing, jewelry, etc.)</li>
              </ul>
            </Box>
            <Box>
              <Typography variant="subtitle1" sx={{ fontSize: '0.95rem', mb: 1 }}>Optional: Temporal Tracking</Typography>
              <Typography variant="body2" sx={{ color: '#666' }}>
                If you would like to enable temporal tracking with size analysis, place a penny next to the skin area. Ensure the penny does not cover any part of the condition, and keep it on the same surface/plane as the skin so the size reference is accurate.
              </Typography>
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

            {error && (
              <Box sx={{
                bgcolor: '#fff3e0',
                border: '1px solid #ff9800',
                borderRadius: 1,
                p: 2,
                mb: 2
              }}>
                <Typography variant="body2" sx={{ color: '#e65100', fontWeight: 500 }}>
                  {error}
                </Typography>
              </Box>
            )}

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
                    <Button size="small" variant="text" onClick={() => { setFile(null); setPreview(null); setError(null); }}>Clear image</Button>
                  </Box>
                </Box>
              )}
            </Box>
          </Box>
        )}

        {/* Step 3: Crop image */}
        {step === 3 && preview && (
          <ImageCropper
            imageSrc={preview}
            onCropComplete={handleCropComplete}
            onCancel={() => setStep(2)}
          />
        )}


        {/* Step 4: Notes */}
        {step === 4 && (
          <Box>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>Optional Notes</Typography>
            <Typography variant="body2" sx={{ color: '#666', mb: 2 }}>Add any symptoms or context (optional, max 1000 characters).</Typography>
            <TextField value={note} onChange={(e) => setNote(e.target.value.slice(0, 1000))} placeholder="Describe symptoms, duration, etc." multiline minRows={4} fullWidth />

            {/* Checkbox for coin presence */}
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
              <input
                type="checkbox"
                id="has-coin-checkbox"
                checked={hasCoin}
                onChange={(e) => setHasCoin(e.target.checked)}
                style={{ marginRight: 8 }}
              />
              <label htmlFor="has-coin-checkbox" style={{ cursor: 'pointer', fontSize: '0.875rem' }}>
                This image contains a coin (for size reference)
              </label>
            </Box>

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
              <Button variant="contained" onClick={() => setStep(3)} disabled={!preview} sx={{ ml: 1 }}>Next</Button>
            </Box>
          </Box>
        )}

        {/* Step 3: Cropping - handled by ImageCropper component with its own buttons */}

        {step === 4 && (
          <Box sx={{ display: 'flex', gap: 2, width: '100%', justifyContent: 'space-between' }}>
            <Button variant="text" onClick={() => setStep(3)}>Back</Button>
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
