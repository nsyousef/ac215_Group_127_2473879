'use client';

import { useEffect, useState } from 'react';
import { Box, Card, CardContent, Typography, Button, Stack } from '@mui/material';
// Use standard <img> for local/data URL previews
import FileAdapter from '@/services/adapters/fileAdapter';
import { isElectron } from '@/utils/config';

export default function TimeTrackingPanel({ conditionId }) {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);

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
  }, [conditionId]);

  return (
    <Card id="time-panel" sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ flex: 1, overflow: 'auto' }}>
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

      <Box sx={{ p: 2, borderTop: '1px solid #eee' }}>
        <Button variant="contained" fullWidth sx={{ textTransform: 'none' }}>Add Image</Button>
      </Box>
    </Card>
  );
}
