'use client';

import { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Stack,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import { useProfile } from '@/contexts/ProfileContext';
import FileAdapter from '@/services/adapters/fileAdapter';
import ProfileFields from '@/components/ProfileFields';

// Options are provided within ProfileFields

export default function ProfilePage({ onBack, showActions = true }) {
  const { profile, updateProfile, loading } = useProfile();
  const [formData, setFormData] = useState({
    dateOfBirth: '',
    sex: '',
    raceEthnicity: '',
    country: '',
  });
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [isDateOfBirthValid, setIsDateOfBirthValid] = useState(true);

  // Load profile data into form when component mounts or profile changes
  useEffect(() => {
    if (!loading) {
      setFormData({
        dateOfBirth: profile.dateOfBirth || '',
        sex: profile.sex || '',
        raceEthnicity: profile.raceEthnicity || '',
        country: profile.country || '',
      });
    }
  }, [profile, loading]);

  const handleChange = (field) => (event) => {
    setFormData((prev) => ({ ...prev, [field]: event.target.value }));
  };

  const handleSave = async () => {
    if (!isDateOfBirthValid) return;
    await updateProfile(formData);
    // Navigate back after saving
    if (onBack) onBack();
  };

  const handleOpenConfirm = () => setConfirmOpen(true);
  const handleCloseConfirm = () => setConfirmOpen(false);

  const handleConfirmReset = async () => {
    try {
      // Clear Electron-managed data (diseases, chat, time_tracking, profile)
      await FileAdapter.resetAppData();
    } catch (e) {
      console.warn('Failed to clear Electron data:', e);
    }

    try {
      // Clear Python-managed data (demographics, case directories)
      if (window.electronAPI?.resetPythonData) {
        await window.electronAPI.resetPythonData();
      }
    } catch (e) {
      console.warn('Failed to clear Python data:', e);
    }

    // Reset local profile to defaults and mark onboarding incomplete
    await updateProfile({ dateOfBirth: '', sex: '', raceEthnicity: '', country: '', hasCompletedOnboarding: false });
    setConfirmOpen(false);

    // Reload to ensure all contexts pick up cleared state
    if (typeof window !== 'undefined') {
      window.location.href = '/';
    }
  };

  if (loading) {
    return (
      <Box sx={{ py: 4, textAlign: 'center' }}>
        <Typography variant="body2">Loading profile...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
            Profile Information
          </Typography>

          <Stack spacing={3}>
            <ProfileFields
              value={formData}
              onChange={setFormData}
              onDateValidityChange={setIsDateOfBirthValid}
            />

            {showActions && (
              <>
                <Button
                  variant="contained"
                  onClick={handleSave}
                  fullWidth
                  sx={{ mt: 2 }}
                  disabled={!isDateOfBirthValid}
                >
                  Save Profile
                </Button>

                <Button
                  variant="outlined"
                  color="error"
                  onClick={handleOpenConfirm}
                  fullWidth
                  sx={{ mt: 1 }}
                >
                  Reset App (Clear All Data)
                </Button>
              </>
            )}
          </Stack>
        </CardContent>
      </Card>

      {showActions && (
        <Dialog open={confirmOpen} onClose={handleCloseConfirm}>
          <DialogTitle>Reset App and Clear All Data?</DialogTitle>
          <DialogContent>
            <Typography variant="body2">
              This will remove all saved diseases, time tracking, chat history, and profile settings. You will be returned to the onboarding flow.
            </Typography>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleCloseConfirm}>Cancel</Button>
            <Button color="error" variant="contained" onClick={handleConfirmReset}>Reset</Button>
          </DialogActions>
        </Dialog>
      )}
    </Box>
  );
}
