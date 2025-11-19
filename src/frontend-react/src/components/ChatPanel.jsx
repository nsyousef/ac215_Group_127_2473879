'use client';

import { useEffect, useState, useRef } from 'react';
import { Box, Card, CardContent, Typography, TextField, IconButton, useTheme, CircularProgress } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import FileAdapter from '@/services/adapters/fileAdapter';
import { isElectron } from '@/utils/config';
import { useDiseaseContext } from '@/contexts/DiseaseContext';
import mlClient from '@/services/mlClient';

export default function ChatPanel({ conditionId }) {
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const listRef = useRef(null);
  const { diseases } = useDiseaseContext();

  const theme = useTheme();
  const primary = theme?.palette?.primary?.main || '#0891B2';
  const primaryContrast = theme?.palette?.primary?.contrastText || '#fff';

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        let data = [];

        // Find the selected condition to check for initial LLM response
        const condition = (diseases || []).find((d) => d.id === conditionId);

        // In Electron, try to load from FileAdapter (file system); else start empty
        if (isElectron() && conditionId) {
          try {
            data = await FileAdapter.loadChat(conditionId);
          } catch (e) {
            console.warn('FileAdapter failed to load chat history', e);
            data = [];
          }
        } else if (!isElectron()) {
          // Not in Electron, do not load bundled data
          data = [];
        }

        // Filter by conditionId if provided
        const filtered = (data || []).filter((m) => {
          if (!conditionId) return false;
          return m.conditionId ? m.conditionId === conditionId : false;
        });

        // If no chat history but condition has initial LLM response, show it
        if (filtered.length === 0 && condition?.llmResponse) {
          filtered.push({
            id: `initial_${conditionId}`,
            role: 'assistant',
            text: condition.llmResponse,
            time: condition.createdAt || new Date().toISOString(),
            conditionId,
            isInitial: true,
          });
        }

        if (mounted) setMessages(filtered || []);
      } catch (e) {
        console.error('Failed to load chat history', e);
      }
    }
    load();
    return () => (mounted = false);
  }, [conditionId, diseases]);

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages]);

  const send = async () => {
    if (!text.trim()) return;
    if (!conditionId) return; // don't send when no condition selected
    
    // Add user message immediately
    const userMsg = {
      id: `u${Date.now()}`,
      role: 'user',
      text: text.trim(),
      time: new Date().toISOString(),
      conditionId,
    };
    setMessages((s) => [...s, userMsg]);
    setText('');
    setLoading(true);

    try {
      // Find the condition to get the case ID
      const condition = (diseases || []).find((d) => d.id === conditionId);
      const caseId = condition?.caseId || `case_${conditionId}`;

      // Call ML client for follow-up response
      const response = await mlClient.chatMessage(caseId, userMsg.text);

      // Add assistant response
      const assistantMsg = {
        id: `a${Date.now()}`,
        role: 'assistant',
        text: response.answer,
        time: new Date().toISOString(),
        conditionId,
      };
      setMessages((s) => [...s, assistantMsg]);

      // Save conversation to FileAdapter if available
      try {
        if (FileAdapter && FileAdapter.saveChat) {
          const allMessages = [...messages, userMsg, assistantMsg];
          await FileAdapter.saveChat(conditionId, allMessages);
        }
      } catch (e) {
        console.warn('Failed to save chat:', e);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      // Show error message to user
      const errorMsg = {
        id: `e${Date.now()}`,
        role: 'assistant',
        text: 'Sorry, I encountered an error processing your message. Please try again.',
        time: new Date().toISOString(),
        conditionId,
      };
      setMessages((s) => [...s, errorMsg]);
    } finally {
      setLoading(false);
    }
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
              <Box sx={{ bgcolor: m.role === 'user' ? primary : '#eee', color: m.role === 'user' ? primaryContrast : '#000', p: 1.5, borderRadius: 2 }}>
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
          disabled={loading || !conditionId}
          onKeyDown={(e) => { if (e.key === 'Enter' && !loading) send(); }}
        />
        <IconButton
          color="primary"
          onClick={send}
          disabled={loading || !conditionId || !text.trim()}
          sx={{
            bgcolor: primary,
            color: primaryContrast,
            '&:hover': { bgcolor: '#067891' },
            '&:disabled': { bgcolor: '#ccc', color: '#999' },
          }}
        >
          {loading ? <CircularProgress size={24} color="inherit" /> : <SendIcon />}
        </IconButton>
      </Box>
    </Card>
  );
}
