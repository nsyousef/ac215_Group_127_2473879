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
        // Find the selected condition to get case ID
        const condition = (diseases || []).find((d) => d.id === conditionId);
        if (!condition) {
          if (mounted) setMessages([]);
          return;
        }

        const caseId = condition.caseId || `case_${conditionId}`;

        // Always try to load fresh conversation history from Python (source of truth)
        let conversationData = null;
        if (isElectron() && window.electronAPI?.mlLoadConversationHistory) {
          try {
            conversationData = await window.electronAPI.mlLoadConversationHistory(caseId);
          } catch (e) {
            console.error('Failed to load conversation history from Python:', e);
            // Fallback to in-memory conversation history if IPC fails
            conversationData = condition.conversationHistory;
          }
        } else {
          // Not in Electron - use in-memory conversation history
          conversationData = condition.conversationHistory;
        }

        // Convert Python conversation format to UI message format
        // Python format: [{ user: { message, timestamp }, llm: { message, timestamp } }, ...]
        const messages = [];
        (conversationData || []).forEach((entry, idx) => {
          if (entry.user) {
            messages.push({
              id: `u_${idx}`,
              role: 'user',
              text: entry.user.message,
              time: entry.user.timestamp,
              conditionId,
            });
          }
          if (entry.llm) {
            messages.push({
              id: `a_${idx}`,
              role: 'assistant',
              text: entry.llm.message,
              time: entry.llm.timestamp,
              conditionId,
              isInitial: idx === 0, // First LLM response is initial
            });
          }
        });

        // If no conversation history but we have llmResponse, show it as initial message
        if (messages.length === 0 && condition?.llmResponse && mounted) {
          messages.push({
            id: `initial_${conditionId}`,
            role: 'assistant',
            text: condition.llmResponse,
            time: condition.createdAt || condition.date || new Date().toISOString(),
            conditionId,
            isInitial: true,
          });
        }

        if (mounted) setMessages(messages);
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

      // Note: Conversation is already saved by Python in chat_message() method
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
