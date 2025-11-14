'use client';

import { useEffect, useState, useRef } from 'react';
import { Box, Card, CardContent, Typography, Button, Stack, useTheme, useMediaQuery, Tooltip } from '@mui/material';
// Use standard <img> for local/data URL previews
import FileAdapter from '@/services/adapters/fileAdapter';
import { isElectron } from '@/utils/config';

export default function TimeTrackingPanel({ conditionId, onAddImage, refreshKey }) {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const contentRef = useRef(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoading(true);
      try {
        let data = [];

        // In Electron, try to load from FileAdapter (file system); otherwise fall back to bundled JSON
        if (isElectron() && conditionId) {
          try {
            data = await FileAdapter.loadTimeTracking(conditionId);
          } catch (e) {
            console.warn('FileAdapter failed, falling back to bundled data', e);
            // Fall back to bundled JSON
            const res = await fetch('/assets/data/time_tracking.json');
            data = await res.json();
          }
        } else if (!isElectron()) {
          // Not in Electron, load from bundled JSON
          const res = await fetch('/assets/data/time_tracking.json');
          data = await res.json();
        }

        // Filter by conditionId: if conditionId is provided, only show its entries.
        const filtered = (data || []).filter((e) => {
          if (!conditionId) return false; // don't show any entries if no condition selected
          return e.conditionId ? e.conditionId === conditionId : false;
        });
        const sorted = filtered.slice().sort((a, b) => new Date(b.date) - new Date(a.date));
        if (mounted) setEntries(sorted);
      } catch (e) {
        console.error('Failed to load time tracking data', e);
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => (mounted = false);
  }, [conditionId, refreshKey]);

  // Scroll to top when entries update so newest is visible
  useEffect(() => {
    if (contentRef.current) {
      try {
        contentRef.current.scrollTop = 0;
      } catch (e) {
        // ignore
      }
    }
  }, [entries]);

  return (
    <Card id="time-panel" sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent ref={contentRef} sx={{ flex: 1, overflow: 'auto' }}>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Time Tracking</Typography>
        {!conditionId && (
          <Typography variant="body2" sx={{ color: '#999', textAlign: 'center', py: 4 }}>
            Select a condition to view its time tracking images
          </Typography>
        )}
        <Stack spacing={2}>
          {loading && <Typography variant="body2">Loading...</Typography>}
          {!loading && conditionId && entries.length === 0 && (
            <Typography variant="body2" sx={{ color: '#999' }}>No tracking images for this condition</Typography>
          )}

          {entries.map((entry) => (
            <Box key={entry.id} sx={{ bgcolor: '#fff', borderRadius: 1, overflow: 'hidden', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
              <Box sx={{ width: '100%', height: 180, overflow: 'hidden' }}>
                <img src={entry.image} alt={entry.id} style={{ width: '100%', height: '180px', objectFit: 'cover' }} />
              </Box>
              <CardContent>
                <Typography variant="caption" sx={{ color: '#666' }}>{new Date(entry.date).toDateString()}</Typography>
                <Typography variant="body2" sx={{ mt: 1 }}>{entry.note}</Typography>
              </CardContent>
            </Box>
          ))}
        </Stack>
      </CardContent>

      {/* Fixed Add Image button: appears above bottom nav on mobile and fixed bottom on desktop */}
      <Box
        sx={{
          position: 'fixed',
          right: isMobile ? 16 : '50%',
          bottom: isMobile ? 80 : 24,
          transform: isMobile ? 'none' : 'translateX(50%)',
          width: isMobile ? '56px' : '280px',
          zIndex: 1400,
          display: 'flex',
          alignItems: 'center',
          justifyContent: isMobile ? 'center' : 'stretch',
        }}
      >
        <Tooltip title="Select a condition first" disableHoverListener={!!conditionId}>
          <span>
            <Button
              variant="contained"
              onClick={() => onAddImage && onAddImage(conditionId)}
              fullWidth={!isMobile}
              disabled={!conditionId}
              sx={{
                textTransform: 'none',
                py: isMobile ? 0 : 1.2,
                borderRadius: isMobile ? '50%' : 2,
                width: isMobile ? '56px' : '100%',
                height: isMobile ? '56px' : 'auto',
                minWidth: isMobile ? '56px' : 'auto',
                padding: isMobile ? 0 : undefined,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: isMobile ? 24 : 'inherit',
              }}
            >
              {isMobile ? '+' : 'Add Image'}
            </Button>
          </span>
        </Tooltip>
      </Box>
    </Card>
  );
}
