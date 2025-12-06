'use client';

import { useEffect, useState } from 'react';
import { Box, CircularProgress, Typography, Container } from '@mui/material';

export default function InitializingScreen() {
  const [isVisible, setIsVisible] = useState(true);
  const [fadeOut, setFadeOut] = useState(false);

  useEffect(() => {
    // Auto-hide after 1 second since no ML initialization happens until user clicks "Start Analysis"
    // This gives just enough visual feedback without unnecessary delay
    const timer = setTimeout(() => {
      setFadeOut(true);
      // Remove from DOM after fade-out completes
      setTimeout(() => {
        setIsVisible(false);
      }, 300);
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  if (!isVisible) return null;

  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#0891b2',
        zIndex: 9999,
        opacity: fadeOut ? 0 : 1,
        transition: 'opacity 0.3s ease-in-out',
      }}
    >
      <Container maxWidth="sm">
        <Box sx={{ textAlign: 'center', color: 'white' }}>
          <Typography
            variant="h4"
            sx={{
              marginBottom: 3,
              fontWeight: 600,
              letterSpacing: '-0.5px',
            }}
          >
            pibu.ai
          </Typography>
          <CircularProgress
            sx={{
              marginBottom: 3,
              color: 'white',
            }}
            size={60}
          />
          <Typography
            sx={{
              fontSize: '14px',
              opacity: 0.9,
              fontWeight: 500,
              letterSpacing: '0.3px',
            }}
          >
            Loading...
          </Typography>
        </Box>
      </Container>
    </Box>
  );
}
