'use client';

import { useState, useEffect } from 'react';
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
  'head',
  'torso',
  'left upper arm',
  'left lower arm',
  'right upper arm',
  'right lower arm',
  'left hand',
  'right hand',
  'left upper leg',
  'right upper leg',
  'left lower leg',
  'right lower leg',
  'left foot',
  'right foot',
];

// Format body part name for display (capitalize properly)
function formatBodyPartLabel(bodyPart) {
  return bodyPart
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export default function AddDiseaseFlow({ open, onClose, onSaved, canCancel = true, onboardingBack }) {
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

  useEffect(() => {
    if (!open) reset();
  }, [open]);

  function reset() {
    setStep(0);
    setBodyPart('');
    setFile(null);
    setPreview(null);
    setNote('');
    setAnalyzing(false);
    setError(null);
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
    setAnalyzing(true);
    setError(null);
    try {
      // Ensure we have map coordinates: if user picked from list only, set default based on label
      let finalMapPos = mapPos;
      if (!finalMapPos && bodyPart) {
        if (BODY_PART_DEFAULTS[bodyPart]) {
          const defaults = BODY_PART_DEFAULTS[bodyPart];
          finalMapPos = {
            leftPct: defaults.leftPct,
            topPct: defaults.topPct,
          };
          setMapPos(finalMapPos);
        }
      }
      
      // Generate case ID for this analysis
      const timestamp = Date.now();
      const diseaseId = `${timestamp}`;  // Disease ID (no prefix, just timestamp)
      const caseId = `case_${timestamp}`;  // Folder name for Python storage

      // Save uploaded file to temp location for Python to access via IPC
      let imagePath;
      if (window.electronAPI && window.electronAPI.saveUploadedImage) {
        // Use IPC to save file in main process
        const buffer = await file.arrayBuffer();
        const uint8Array = new Uint8Array(buffer);
        imagePath = await window.electronAPI.saveUploadedImage(caseId, file.name, uint8Array);
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

      // Step 2: Call ML client to get predictions (matches api_manager.py workflow)
      const results = await mlClient.getInitialPrediction(
        imagePath, // file path
        note || '', // text description
        caseId
      );

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

      // Get the enriched disease object from Python response (includes all UI fields)
      // Python's _build_enriched_disease() returns: id, name, description, bodyPart, 
      // mapPosition, image, confidenceLevel, llmResponse, timelineData, conversationHistory
      const enrichedDisease = results.enriched_disease || {};
      
      // Build complete disease object with both minimal data (for diseases.json) 
      // and enriched data (for immediate UI display)
      const newDisease = {
        // Minimal fields (saved to diseases.json)
        id: diseaseId,  // Use timestamp without "case_" prefix
        name: formattedName,
        image: preview,
        // Enriched fields (from case_history.json, already loaded by Python)
        description: enrichedDisease.description || '',
        bodyPart: enrichedDisease.bodyPart || bodyPart || 'Unknown',
        mapPosition: enrichedDisease.mapPosition || null,
        confidenceLevel: enrichedDisease.confidenceLevel || 0,
        date: enrichedDisease.date || '',
        llmResponse: enrichedDisease.llmResponse || '',
        timelineData: enrichedDisease.timelineData || [],
        conversationHistory: enrichedDisease.conversationHistory || []
      };

      // Add to context (saves minimal fields to diseases.json)
      await addDisease(newDisease);

      setAnalyzing(false);
      // Pass complete enriched object to parent so it can be immediately used
      if (onSaved) onSaved(newDisease);
    } catch (e) {
      console.error('Analysis failed:', e);
      setAnalyzing(false);
      setError('Analysis failed. Please try again.');
      // allow user to retry by going back to photo step
      setStep(2);
    }
  };

  const handleStartAnalysis = () => {
    // Move from notes (4) to analyzing (5)
    setStep(5);
    analyzeImage();
  };

  const close = () => {
    reset();
    onClose && onClose();
  };

  const handleDialogClose = (event, reason) => {
    if (!canCancel && (reason === 'backdropClick' || reason === 'escapeKeyDown')) {
      return; // prevent closing during onboarding
    }
    close();
  };

  return (
    <Dialog open={open} fullScreen={fullScreen} onClose={handleDialogClose} disableEscapeKeyDown={!canCancel} fullWidth maxWidth="sm">
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h6" component="div">Add Condition</Typography>
        {canCancel && (
          <IconButton onClick={close} size="small"><CloseIcon /></IconButton>
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
          </Box>
        )}

        {/* Step 5: Analyzing */}
        {step === 5 && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, py: 4 }}>
            <Typography variant="h6">Analyzing...</Typography>
            <Typography variant="body2" sx={{ color: '#666', textAlign: 'center' }}>We never send your raw image to the server. Only a private embedding is generated for analysis.</Typography>
            <Box sx={{ width: '80%', mt: 2 }}>
              <LinearProgress sx={{ mt: 2 }} />
            </Box>
            {error && <Typography variant="body2" sx={{ color: 'error.main' }}>{error}</Typography>}
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
              <Button variant="contained" onClick={handleStartAnalysis} sx={{ ml: 1 }}>Analyze</Button>
            </Box>
          </Box>
        )}

        {step === 5 && (
          <Box sx={{ display: 'flex', gap: 2, width: '100%', justifyContent: 'center' }}>
            {canCancel && <Button variant="text" onClick={close}>Cancel</Button>}
          </Box>
        )}
      </DialogActions>
    </Dialog>
  );
}
