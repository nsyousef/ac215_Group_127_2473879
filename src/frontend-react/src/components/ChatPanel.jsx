'use client';

import { useEffect, useState, useRef } from 'react';
import { Box, Card, CardContent, Typography, TextField, IconButton } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import FileAdapter from '@/services/adapters/fileAdapter';
import { isElectron } from '@/utils/config';

export default function ChatPanel({ conditionId }) {
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState('');
  const listRef = useRef(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        let data = [];

        // In Electron, try to load from FileAdapter (file system); otherwise fall back to bundled JSON
        if (isElectron() && conditionId) {
          try {
            data = await FileAdapter.loadChat(conditionId);
          } catch (e) {
            console.warn('FileAdapter failed, falling back to bundled data', e);
            // Fall back to bundled JSON
            const res = await fetch('/assets/data/chat_history.json');
            data = await res.json();
          }
        } else if (!isElectron()) {
          // Not in Electron, load from bundled JSON
          const res = await fetch('/assets/data/chat_history.json');
          data = await res.json();
        }

        // Filter by conditionId if provided. If no conditionId, show nothing and ask user to select.
        const filtered = (data || []).filter((m) => {
          if (!conditionId) return false;
          return m.conditionId ? m.conditionId === conditionId : false;
        });
        if (mounted) setMessages(filtered || []);
      } catch (e) {
        console.error('Failed to load chat history', e);
      }
    }
    load();
    return () => (mounted = false);
  }, [conditionId]);

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages]);

  const send = () => {
    if (!text.trim()) return;
    if (!conditionId) return; // don't send when no condition selected
    const m = { id: `u${Date.now()}`, role: 'user', text: text.trim(), time: new Date().toISOString(), conditionId };
    setMessages((s) => [...s, m]);
    setText('');
    // placeholder assistant reply
    setTimeout(() => {
      setMessages((s) => [...s, { id: `a${Date.now()}`, role: 'assistant', text: 'Thanks — I will look into that (placeholder response).', time: new Date().toISOString(), conditionId }]);
    }, 700);
  };

  return (
    <Card id="chat-panel" sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ flex: 1, overflow: 'auto', p: 2 }} ref={listRef}>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Chat with Pibu</Typography>

        {!conditionId && (
          <Typography variant="body2" sx={{ color: '#999', textAlign: 'center', py: 4 }}>
            Select a condition to view its chat history
          </Typography>
        )}

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          {messages.map((m) => (
            <Box key={m.id} sx={{ alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '80%' }}>
              <Box sx={{ bgcolor: m.role === 'user' ? '#1976d2' : '#eee', color: m.role === 'user' ? '#fff' : '#000', p: 1.5, borderRadius: 2 }}>
                <Typography variant="body2">{m.text}</Typography>
              </Box>
              <Typography variant="caption" sx={{ color: '#999' }}>{new Date(m.time).toLocaleString()}</Typography>
            </Box>
          ))}
        </Box>
      </CardContent>

      <Box sx={{ p: 1, borderTop: '1px solid #eee', display: 'flex', gap: 1 }}>
        <TextField
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={conditionId ? 'Write a message' : 'Select a condition to send messages'}
          fullWidth
          size="small"
          onKeyDown={(e) => { if (e.key === 'Enter') send(); }}
        />
        <IconButton color="primary" onClick={send} sx={{ bgcolor: '#1976d2', color: '#fff' }}>
          <SendIcon />
        </IconButton>
      </Box>
    </Card>
  );
}
