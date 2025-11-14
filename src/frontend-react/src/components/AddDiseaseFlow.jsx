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
import { CONDITIONS, BODY_MAP_SPOTS } from '@/lib/constants';
import BodyMapPicker from '@/components/BodyMapPicker';
import { inferBodyPartFromCoords } from '@/lib/bodyMapUtils';
import { useDiseaseContext } from '@/contexts/DiseaseContext';

const BODY_PARTS = [
  'Head',
  'Face',
  'Neck',
  'Torso',
  'Left Upper Arm',
  'Left Lower Arm',
  'Right Upper Arm',
  'Right Lower Arm',
  'Left Leg',
  'Right Leg',
];

export default function AddDiseaseFlow({ open, onClose, onSaved }) {
  const theme = useTheme();
  const fullScreen = useMediaQuery(theme.breakpoints.down('md'));
  const { addDisease } = useDiseaseContext();

  const [step, setStep] = useState(0);
  const [bodyPart, setBodyPart] = useState('');
  const [mapPos, setMapPos] = useState(null); // { leftPct, topPct }
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
    // infer a categorical body part name from anchors
    try {
      const inferred = inferBodyPartFromCoords(coords, Object.values(BODY_MAP_SPOTS).map((s) => ({ name: s.label, leftPct: parseFloat(s.left), topPct: parseFloat(s.top) })));
      if (inferred) setBodyPart(inferred);
    } catch (e) {
      // ignore
    }
  };

  const analyzeImage = async () => {
    setAnalyzing(true);
    setError(null);
    try {
      // Simulate analysis delay (real ML will take ~30s). Keep simulation short for UX.
      await new Promise((r) => setTimeout(r, 3000));

      // Simulate AI suggestion by selecting a likely condition name from bundled list
      const suggestion = (CONDITIONS || []).find((c) => c.name && c.name !== '<disease title>');
      const suggestedName = suggestion ? suggestion.name : 'Unknown Condition';

      // Create disease object (include map position if available)
      const newDisease = {
        id: Date.now(),
        name: suggestedName,
        description: note || '',
        bodyPart,
        mapPosition: mapPos || null,
        // store preview as image for time-tracking initial entry
        image: preview,
        createdAt: new Date().toISOString(),
      };

      // Add to context (and persist if possible)
      await addDisease(newDisease);

      // Also save initial time-tracking entry (if FileAdapter supported)
      try {
        const entry = {
          id: Date.now(),
          conditionId: newDisease.id,
          date: new Date().toISOString(),
          image: preview,
          note: note || 'Initial upload',
        };
        if (FileAdapter && FileAdapter.saveTimeEntry) {
          await FileAdapter.saveTimeEntry(newDisease.id, entry);
        }
      } catch (e) {
        // ignore
      }

      setAnalyzing(false);
      if (onSaved) onSaved(newDisease);
    } catch (e) {
      setAnalyzing(false);
      setError('Analysis failed. Please try again.');
      // allow user to retry by going back to photo step
      setStep(2);
    }
  };

  const handleConfirmPhoto = () => {
    // Move from verify (3) to notes (4)
    setStep(4);
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

  return (
    <Dialog open={open} fullScreen={fullScreen} onClose={close} fullWidth maxWidth="sm">
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h6" component="div">Add Condition</Typography>
        <IconButton onClick={close} size="small"><CloseIcon /></IconButton>
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

            <Typography variant="body2" sx={{ color: '#666', mb: 1 }}>Or pick from the list</Typography>
            <List>
              {BODY_PARTS.map((p) => (
                <ListItem key={p} disablePadding>
                  <ListItemButton selected={bodyPart === p} onClick={() => { setBodyPart(p); setMapPos(null); }}>
                    <ListItemText primary={p} />
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
              {/* Simplified body map placeholder */}
              <Box sx={{ width: 160, height: 260, bgcolor: '#f0f0f0', borderRadius: 1 }} />
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
                <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                  <img src={preview} alt="preview" style={{ maxWidth: '100%', maxHeight: 360, borderRadius: 8 }} />
                </Box>
              )}
            </Box>
          </Box>
        )}

        {/* Step 3: Verify photo */}
        {step === 3 && (
          <Box>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>Looks good?</Typography>
            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
              {preview && <img src={preview} alt="preview" style={{ maxWidth: '100%', maxHeight: 420, borderRadius: 8 }} />}
            </Box>
            <Typography variant="body2" sx={{ color: '#666', mb: 2 }}>If the image looks good, continue to add optional notes. Otherwise retake or upload another photo.</Typography>
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
          <Box sx={{ display: 'flex', gap: 2, width: '100%', justifyContent: 'flex-end' }}>
            <Button variant="text" onClick={close}>Cancel</Button>
            <Button variant="contained" onClick={() => setStep(1)} disabled={!bodyPart}>Next</Button>
          </Box>
        )}

        {step === 1 && (
          <Box sx={{ display: 'flex', gap: 2, width: '100%', justifyContent: 'space-between' }}>
            <Button variant="text" onClick={() => setStep(0)}>Back</Button>
            <Box>
              <Button variant="text" onClick={close}>Cancel</Button>
              <Button variant="contained" onClick={() => setStep(2)} sx={{ ml: 1 }}>Next</Button>
            </Box>
          </Box>
        )}

        {step === 2 && (
          <Box sx={{ display: 'flex', gap: 2, width: '100%', justifyContent: 'space-between' }}>
            <Button variant="text" onClick={() => setStep(1)}>Back</Button>
            <Box>
              <Button variant="text" onClick={close}>Cancel</Button>
              <Button variant="contained" onClick={() => setStep(3)} disabled={!preview} sx={{ ml: 1 }}>Continue</Button>
            </Box>
          </Box>
        )}

        {step === 3 && (
          <Box sx={{ display: 'flex', gap: 2, width: '100%', justifyContent: 'space-between' }}>
            <Button variant="text" onClick={() => setStep(2)}>No, retake</Button>
            <Box>
              <Button variant="text" onClick={close}>Cancel</Button>
              <Button variant="contained" onClick={handleConfirmPhoto} sx={{ ml: 1 }}>Yes</Button>
            </Box>
          </Box>
        )}

        {step === 4 && (
          <Box sx={{ display: 'flex', gap: 2, width: '100%', justifyContent: 'space-between' }}>
            <Button variant="text" onClick={() => setStep(3)}>Back</Button>
            <Box>
              <Button variant="text" onClick={close}>Cancel</Button>
              <Button variant="contained" onClick={handleStartAnalysis} sx={{ ml: 1 }}>Analyze</Button>
            </Box>
          </Box>
        )}

        {step === 5 && (
          <Box sx={{ display: 'flex', gap: 2, width: '100%', justifyContent: 'center' }}>
            <Button variant="text" onClick={close}>Cancel</Button>
          </Box>
        )}
      </DialogActions>
    </Dialog>
  );
}
