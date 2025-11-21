'use client';

import { useEffect, useState, useRef } from 'react';
import { Box, Card, CardContent, Typography, Button, Stack, useTheme, useMediaQuery, Tooltip } from '@mui/material';
// Use standard <img> for local/data URL previews
import { isElectron } from '@/utils/config';
import { useDiseaseContext } from '@/contexts/DiseaseContext';

export default function TimeTrackingPanel({ conditionId, onAddImage, refreshKey }) {
  const { diseases } = useDiseaseContext();
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

        // In Electron, try to load timeline data
        if (isElectron() && conditionId) {
          try {
            // Get case ID from condition
            const condition = (diseases || []).find((d) => d.id === conditionId);
            const caseId = condition?.caseId || `case_${conditionId}`;
            
            // Always try to load fresh timeline data from Python (source of truth)
            if (window.electronAPI?.loadCaseHistoryFromPython) {
              try {
                const caseHistory = await window.electronAPI.loadCaseHistoryFromPython(caseId);
                
                // Convert dates object to array format for timeline display
                // dates structure can be:
                // - Simple date: "2025-11-21" (initial ML prediction entry)
                // - Date with timestamp: "2025-11-21_1763762693117" (manual timeline entries)
                const entries = Object.entries(caseHistory.dates || {})
                  .map(([dateKey, entry]) => {
                    // Parse the date key to extract date and timestamp
                    let displayDate, sortTimestamp;
                    if (dateKey.includes('_')) {
                      // Format: "2025-11-21_1763762693117"
                      const [datePart, timestampPart] = dateKey.split('_');
                      displayDate = datePart;
                      sortTimestamp = parseInt(timestampPart, 10);
                    } else {
                      // Format: "2025-11-21" (use date as timestamp for sorting)
                      displayDate = dateKey;
                      sortTimestamp = new Date(dateKey).getTime();
                    }
                    
                    return {
                      id: dateKey, // Use full key as unique ID
                      conditionId: conditionId,
                      date: displayDate, // Clean ISO date string "2025-11-21"
                      timestamp: sortTimestamp, // Numeric timestamp for sorting
                      image: entry.image_path || '',
                      note: entry.text_summary || '', // Display text_summary as notes
                      predictions: entry.predictions || {},
                      cv_analysis: entry.cv_analysis || {}
                    };
                  });
                
                // Convert file paths to data URLs for display in renderer
                data = await Promise.all(
                  entries.map(async (entry) => {
                    if (entry.image && window.electronAPI?.readImageAsDataUrl) {
                      try {
                        const dataUrl = await window.electronAPI.readImageAsDataUrl(entry.image);
                        return { ...entry, image: dataUrl };
                      } catch (e) {
                        console.warn('Failed to load image:', entry.image, e);
                        return entry; // Keep file path as fallback
                      }
                    }
                    return entry;
                  })
                );
              } catch (e) {
                console.warn('Failed to load case history from Python via IPC, using in-memory data', e);
                // Fallback: Use in-memory timelineData if IPC fails
                if (condition?.timelineData && Array.isArray(condition.timelineData)) {
                  data = await Promise.all(
                    condition.timelineData.map(async (entry) => {
                      if (entry.image && window.electronAPI?.readImageAsDataUrl) {
                        try {
                          const dataUrl = await window.electronAPI.readImageAsDataUrl(entry.image);
                          return { ...entry, image: dataUrl, conditionId };
                        } catch (e) {
                          console.warn('Failed to load image:', entry.image, e);
                          return { ...entry, conditionId };
                        }
                      }
                      return { ...entry, conditionId };
                    })
                  );
                }
              }
            } else {
              // No IPC available - use in-memory timelineData
              if (condition?.timelineData && Array.isArray(condition.timelineData)) {
                data = condition.timelineData.map(entry => ({ ...entry, conditionId }));
              }
            }
          } catch (e) {
            console.warn('Failed to load timeline data', e);
            data = [];
          }
        } else if (!isElectron()) {
          // Not in Electron, do not load bundled data
          data = [];
        }

        // Data is already filtered by caseId (from case_history.json)
        // Sort by timestamp descending (most recent first)
        const sorted = (data || []).slice().sort((a, b) => {
          const timestampA = a.timestamp || new Date(a.date).getTime() || 0;
          const timestampB = b.timestamp || new Date(b.date).getTime() || 0;
          return timestampB - timestampA;
        });
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
