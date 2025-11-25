'use client';

import { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  TextField,
  IconButton,
  LinearProgress,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useDiseaseContext } from '@/contexts/DiseaseContext';

export default function AddTimeEntryFlow({ open, onClose, conditionId, onSaved }) {
  const { diseases } = useDiseaseContext();
  const theme = useTheme();
  const fullScreen = useMediaQuery(theme.breakpoints.down('md'));

  const [step, setStep] = useState(1); // 1: instructions, 2: photo, 4: notes, 5: analyzing
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [note, setNote] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);

  const reset = () => {
    setStep(1);
    setFile(null);
    setPreview(null);
    setNote('');
    setAnalyzing(false);
    setError(null);
  };

  const close = () => {
    reset();
    onClose && onClose();
  };

  const onFileSelected = (f) => {
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result);
    reader.readAsDataURL(f);
    setFile(f);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) onFileSelected(f);
  };

  const analyze = async () => {
    if (!conditionId) return setError('No condition selected');
    if (!file) return setError('No image selected');
    
    setAnalyzing(true);
    setError(null);
    
    try {
      // Find the case ID from the condition
      const condition = diseases.find(d => d.id === conditionId);
      const caseId = condition?.caseId || `case_${conditionId}`;
      
      // Step 1: Save the image file to disk via IPC
      let imagePath;
      if (window.electronAPI?.saveUploadedImage) {
        const buffer = await file.arrayBuffer();
        const uint8Array = new Uint8Array(buffer);
        const timestamp = Date.now();
        imagePath = await window.electronAPI.saveUploadedImage(caseId, `timeline_${timestamp}_${file.name}`, uint8Array);
      } else {
        throw new Error('Image upload is only available in Electron runtime.');
      }
      
      // Step 2: Add timeline entry to case_history.json via Python
      // Use date + timestamp to ensure uniqueness if multiple uploads on same day
      const now = new Date();
      const dateKey = `${now.toISOString().split('T')[0]}_${Date.now()}`; // "2025-11-21_1763762693117"
      
      if (window.electronAPI?.addTimelineEntry) {
        await window.electronAPI.addTimelineEntry(caseId, imagePath, note || '', dateKey);
      } else {
        throw new Error('Timeline entry save is only available in Electron runtime.');
      }
      
      // Create entry object for callback
      const entry = {
        id: dateKey,
        conditionId,
        date: dateKey,
        image: imagePath,
        note: note || '',
      };

      setAnalyzing(false);
      onSaved && onSaved(entry);
      close();
    } catch (e) {
      console.error('Failed to save timeline entry:', e);
      setAnalyzing(false);
      setError('Failed to save image. Please try again.');
      setStep(2);
    }
  };

  return (
    <Dialog open={open} fullScreen={fullScreen} onClose={close} fullWidth maxWidth="sm">
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h6" component="div">Add Image</Typography>
        <IconButton onClick={close} size="small"><CloseIcon /></IconButton>
      </DialogTitle>

      <DialogContent dividers sx={{ overflow: 'auto', WebkitOverflowScrolling: 'touch' }}>
        {step === 1 && (
          <Box>
            <Typography variant="subtitle1">Image Instructions</Typography>
            <Typography variant="body2" sx={{ color: '#666', mt: 1 }}>
              Place a coin or similarly sized object near the skin condition. Take a clear photo.
            </Typography>
            <Box sx={{ my: 2, display: 'flex', justifyContent: 'center' }}>
              <img src="/assets/instructions.png" alt="Coin placement instructions" style={{ maxWidth: 320, width: '100%', borderRadius: 8, boxShadow: '0 2px 8px #0001' }} />
            </Box>
          </Box>
        )}

        {step === 2 && (
          <Box>
            <Typography variant="subtitle1">Take or Upload Photo</Typography>
            <Box onDragOver={(e) => e.preventDefault()} onDrop={handleDrop} sx={{ border: '2px dashed #ddd', borderRadius: 1, p: 2, textAlign: 'center', mt: 1 }}>
              {!preview ? (
                <>
                  <input id="time-file-input" type="file" accept="image/*" capture="environment" style={{ display: 'none' }} onChange={(e) => onFileSelected(e.target.files[0])} />
                  <label htmlFor="time-file-input">
                    <Button component="span" variant="outlined">Choose file or take photo</Button>
                  </label>
                  <Typography variant="caption" sx={{ display: 'block', mt: 1 }}>Or drag & drop an image here</Typography>
                </>
              ) : (
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                    <img src={preview} alt="preview" style={{ maxWidth: '100%', maxHeight: 420, borderRadius: 8 }} />
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
                    <Button size="small" variant="text" onClick={() => { setFile(null); setPreview(null); }}>Clear image</Button>
                  </Box>
                </Box>
              )}
            </Box>
          </Box>
        )}

        {step === 4 && (
          <Box>
            <Typography variant="subtitle1">Optional Notes</Typography>
            <TextField value={note} onChange={(e) => setNote(e.target.value.slice(0, 1000))} placeholder="Describe symptoms, duration, etc." multiline minRows={4} fullWidth sx={{ mt: 1 }} />
          </Box>
        )}

        {step === 5 && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, py: 4 }}>
            <Typography variant="h6">Saving...</Typography>
            <Typography variant="body2" sx={{ color: '#666', textAlign: 'center' }}>Saving image to timeline. This may take a moment.</Typography>
            <Box sx={{ width: '80%', mt: 2 }}>
              <LinearProgress />
            </Box>
            {error && <Typography variant="body2" sx={{ color: 'error.main' }}>{error}</Typography>}
          </Box>
        )}
      </DialogContent>

      <DialogActions sx={{ px: 3, py: 2 }}>
        {step === 1 && (
          <Box sx={{ width: '100%', display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
            <Button variant="text" onClick={close}>Cancel</Button>
            <Button variant="contained" onClick={() => setStep(2)}>Next</Button>
          </Box>
        )}

        {step === 2 && (
          <Box sx={{ width: '100%', display: 'flex', justifyContent: 'space-between' }}>
            <Button variant="text" onClick={() => setStep(1)}>Back</Button>
            <Box>
              <Button variant="text" onClick={close}>Cancel</Button>
              <Button variant="contained" onClick={() => setStep(4)} disabled={!preview} sx={{ ml: 1 }}>Continue</Button>
            </Box>
          </Box>
        )}

        {step === 4 && (
          <Box sx={{ width: '100%', display: 'flex', justifyContent: 'space-between' }}>
            <Button variant="text" onClick={() => setStep(2)}>Back</Button>
            <Box>
              <Button variant="text" onClick={close}>Cancel</Button>
              <Button variant="contained" onClick={() => { setStep(5); analyze(); }} sx={{ ml: 1 }}>Save</Button>
            </Box>
          </Box>
        )}

        {step === 5 && (
          <Box sx={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
            <Button variant="text" onClick={close}>Cancel</Button>
          </Box>
        )}
      </DialogActions>
    </Dialog>
  );
}
