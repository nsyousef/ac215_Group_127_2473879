'use client';

import { useState, useEffect } from 'react';
import { useTheme, useMediaQuery, Box, Button, Typography, Card, CardContent, Stack, Link } from '@mui/material';
import AddDiseaseFlow from '@/components/AddDiseaseFlow';
import ProfileFields from '@/components/ProfileFields';
import { useProfile } from '@/contexts/ProfileContext';

function openExternal(url) {
  if (typeof window !== 'undefined' && window.electronAPI && window.electronAPI.openExternal) {
    window.electronAPI.openExternal(url);
  } else if (typeof window !== 'undefined') {
    window.open(url, '_blank', 'noopener,noreferrer');
  }
}

export default function OnboardingFlow({ onComplete }) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const { updateProfile, profile } = useProfile();
  const [step, setStep] = useState('splash'); // 'splash' | 'profile' | 'add'
  const [showAddFlow, setShowAddFlow] = useState(false);
  const [formData, setFormData] = useState({ dateOfBirth: '', sex: '', raceEthnicity: '', country: '' });

  useEffect(() => {
    if (profile) {
      setFormData({
        dateOfBirth: profile.dateOfBirth || '',
        sex: profile.sex || '',
        raceEthnicity: profile.raceEthnicity || '',
        country: profile.country || '',
      });
    }
  }, [profile]);

  const handleGetStarted = () => setStep('profile');

  const handleProfileNext = async () => {
    // Persist optional profile data (can be blank)
    await updateProfile({ ...formData });
    setStep('add');
    setShowAddFlow(true);
  };

  const handleAddSaved = async (newDisease) => {
    // mark onboarding complete
    await updateProfile({ hasCompletedOnboarding: true });
    setShowAddFlow(false);
    if (onComplete) onComplete(newDisease);
  };

  if (step === 'splash') {
    return (
      <Box sx={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: '#f5f5f5', p: 2 }}>
        <Card sx={{ maxWidth: 520, width: '100%' }}>
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', py: 3 }}>
              <img src="/assets/pibu_logo.svg" alt="pibu.ai" width={120} height={120} style={{ marginBottom: 16, borderRadius: 8, boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }} />
              <Typography variant="h5" sx={{ fontWeight: 700, mb: 1 }}>Welcome</Typography>
              <Typography variant="body2" sx={{ color: '#666', mb: 3 }}>
                Track your skin conditions over time with helpful insights.
              </Typography>
              <Button variant="contained" size={isMobile ? 'medium' : 'large'} onClick={handleGetStarted}>
                Get Started
              </Button>
              <Typography variant="caption" sx={{ color: '#888', mt: 3 }}>
                By using this app you agree to our{' '}
                <Link component="button" onClick={() => openExternal('https://example.com/terms')}>Terms of Service</Link>
                {' '}and{' '}
                <Link component="button" onClick={() => openExternal('https://example.com/privacy')}>Privacy Policy</Link>.
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </Box>
    );
  }

  if (step === 'profile') {
    return (
      <Box sx={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: '#f5f5f5', p: 2 }}>
        <Card sx={{ maxWidth: 520, width: '100%' }}>
          <CardContent>
            <Typography variant="h6" sx={{ fontWeight: 700, mb: 2 }}>Tell us about you (optional)</Typography>
            <Typography variant="body2" sx={{ color: '#666', mb: 2 }}>This helps personalize recommendations.</Typography>
            <ProfileFields value={formData} onChange={setFormData} />
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
              <Button variant="text" onClick={() => setStep('splash')}>Back</Button>
              <Button variant="contained" onClick={handleProfileNext}>Continue</Button>
            </Box>
          </CardContent>
        </Card>
      </Box>
    );
  }

  // step === 'add'
  return (
    <>
      <AddDiseaseFlow
        open={showAddFlow}
        onClose={() => { /* ignore close during onboarding */ }}
        onSaved={handleAddSaved}
        canCancel={false}
        onboardingBack={() => {
          // Close the add flow and return to profile step during onboarding
          setShowAddFlow(false);
          setStep('profile');
        }}
      />
    </>
  );
}
