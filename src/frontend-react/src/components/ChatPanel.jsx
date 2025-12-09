'use client';

import { useEffect, useState, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  IconButton,
  useTheme,
  CircularProgress,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import { isElectron } from '@/utils/config';
import { useDiseaseContext } from '@/contexts/DiseaseContext';
import mlClient from '@/services/mlClient';

// Helper to convert **bold** markdown to HTML
function markdownToHtml(text) {
  if (!text) return '';
  return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

export default function ChatPanel({ conditionId, refreshKey }) {
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const listRef = useRef(null);
  const { diseases, reload: reloadDiseases } = useDiseaseContext();

  // Tracks which message is currently receiving initial-explain chunks
  const streamingInitialIdRef = useRef(null);

  const theme = useTheme();
  const primary = theme?.palette?.primary?.main || '#0891B2';
  const primaryContrast = theme?.palette?.primary?.contrastText || '#fff';

  // Load conversation history when condition changes or refreshKey changes
  useEffect(() => {
    if (!conditionId) {
      setMessages([]);
      return;
    }

    let mounted = true;
    streamingInitialIdRef.current = null; // reset when condition changes

    // Clear messages immediately when condition changes to show loading state
    setMessages([]);

    async function load() {
      try {
        console.log('[ChatPanel] Loading conversation history for conditionId:', conditionId, 'refreshKey:', refreshKey);
        const condition = (diseases || []).find((d) => d.id === conditionId);
        if (!condition) {
          console.log('[ChatPanel] Condition not found for id:', conditionId);
          if (mounted) setMessages([]);
          return;
        }

        const caseId = condition.caseId || `case_${conditionId}`;
        console.log('[ChatPanel] Using caseId:', caseId);

        // Always try to load fresh conversation history from Python (source of truth)
        let conversationData = null;
        if (isElectron() && window.electronAPI?.mlLoadConversationHistory) {
          try {
            console.log('[ChatPanel] Calling mlLoadConversationHistory for caseId:', caseId);
            // Always reload from Python to get the latest conversation history
            conversationData = await window.electronAPI.mlLoadConversationHistory(caseId);
            console.log('[ChatPanel] Received conversation data from Python:', conversationData);
          } catch (e) {
            console.error('[ChatPanel] Failed to load conversation history from Python:', e);
            // Fallback to in-memory data if Python load fails
            conversationData = condition.conversationHistory || [];
            console.log('[ChatPanel] Using fallback conversation data:', conversationData);
          }
        } else {
          conversationData = condition.conversationHistory || [];
          console.log('[ChatPanel] Not Electron, using in-memory data:', conversationData);
        }

        const msgs = [];
        (conversationData || []).forEach((entry, idx) => {
          if (entry.user) {
            msgs.push({
              id: `u_${idx}`,
              role: 'user',
              text: entry.user.message,
              time: entry.user.timestamp,
              conditionId,
            });
          }
          if (entry.llm) {
            msgs.push({
              id: `a_${idx}`,
              role: 'assistant',
              text: entry.llm.message,
              time: entry.llm.timestamp,
              conditionId,
              isInitial: idx === 0,
            });
          }
        });

        console.log('[ChatPanel] Processed messages:', msgs.length, 'from', conversationData?.length || 0, 'entries');

        // Legacy: if no conversation history but we have llmResponse, show it
        if (msgs.length === 0 && condition?.llmResponse && mounted) {
          console.log('[ChatPanel] No messages found, using legacy llmResponse');
          msgs.push({
            id: `initial_${conditionId}`,
            role: 'assistant',
            text: condition.llmResponse,
            time: condition.createdAt || condition.date || new Date().toISOString(),
            conditionId,
            isInitial: true,
          });
        }

        if (mounted) {
          console.log('[ChatPanel] Setting messages:', msgs.length);
          setMessages(msgs);
        }
      } catch (e) {
        console.error('Failed to load chat history', e);
      }
    }

    load();
    return () => {
      mounted = false;
    };
  }, [conditionId, diseases, refreshKey]);

  // Auto-scroll on new messages
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages]);

  // === STREAMING INITIAL EXPLANATION (first LLM call) ===
  useEffect(() => {
    if (!isElectron() || !window.electronAPI?.mlOnStreamChunk) {
      return undefined;
    }

    // Subscribe to initial-explain chunks (from ml:getInitialPrediction / predict)
    const unsubscribe = window.electronAPI.mlOnStreamChunk((chunk) => {
      if (!conditionId || !chunk) return;

      setMessages((prev) => {
        // If we already have a streaming-initial message, append to it
        const existingId = streamingInitialIdRef.current;
        if (existingId) {
          return prev.map((m) =>
            m.id === existingId
              ? { ...m, text: (m.text || '') + chunk }
              : m
          );
        }

        // Otherwise, create a new assistant message for this initial stream
        const id = `init_stream_${Date.now()}`;
        streamingInitialIdRef.current = id;

        const newMsg = {
          id,
          role: 'assistant',
          text: chunk,
          time: new Date().toISOString(),
          conditionId,
          isInitial: true,
        };

        return [...prev, newMsg];
      });
    });

    return () => {
      streamingInitialIdRef.current = null;
      if (typeof unsubscribe === 'function') {
        unsubscribe();
      }
    };
  }, [conditionId]);

  // === FOLLOW-UP CHAT (already streaming via mlClient.chatMessageStream) ===
  const send = async () => {
    const userMessageText = text.trim();
    if (!userMessageText) return;

    setText('');
    setLoading(true);

    const timestamp = Date.now();
    const userMsg = {
      id: `u${timestamp}`,
      role: 'user',
      text: userMessageText,
      time: new Date().toISOString(),
      conditionId,
    };

    const assistantId = `a${timestamp}`;
    const assistantMsg = {
      id: assistantId,
      role: 'assistant',
      text: '',
      time: new Date().toISOString(),
      conditionId,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    try {
      const condition = (diseases || []).find((d) => d.id === conditionId);
      const caseId = condition?.caseId || `case_${conditionId}`;

      for await (const chunk of mlClient.chatMessageStream(caseId, userMessageText)) {
        const delta =
          chunk?.delta ??
          chunk?.text ??
          chunk?.answer ??
          chunk ??
          '';

        if (!delta) continue;

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, text: (m.text || '') + delta } : m
          )
        );
      }

      // After message is successfully sent, reload diseases to update lastMessageTimestamp for sorting
      try {
        await reloadDiseases();
        console.log('[ChatPanel] Reloaded diseases after sending message');
      } catch (reloadError) {
        console.warn('[ChatPanel] Failed to reload diseases after message:', reloadError);
      }
    } catch (error) {
      console.error('Error in streaming chat:', error);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId && !m.text
            ? { ...m, text: 'Error getting response from assistant.' }
            : m
        )
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card
      id="chat-panel"
      sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}
    >
      <CardContent sx={{ flex: 1, overflow: 'auto', p: 2 }} ref={listRef}>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          Chat with Pibu
        </Typography>

        {!conditionId && (
          <Typography
            variant="body2"
            sx={{ color: '#999', textAlign: 'center', py: 4 }}
          >
            Select a condition to view its chat history
          </Typography>
        )}

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          {messages.map((m) => (
            <Box
              key={m.id}
              sx={{
                alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                maxWidth: '80%',
              }}
            >
              <Box
                sx={{
                  bgcolor: m.role === 'user' ? primary : '#eee',
                  color: m.role === 'user' ? primaryContrast : '#000',
                  p: 1.5,
                  borderRadius: 2,
                }}
              >
                <Typography variant="body2" dangerouslySetInnerHTML={{ __html: markdownToHtml(m.text) }} />
              </Box>
              <Typography variant="caption" sx={{ color: '#999' }}>
                {new Date(m.time).toLocaleString()}
              </Typography>
            </Box>
          ))}
        </Box>
      </CardContent>

      <Box
        sx={{
          p: 1,
          borderTop: '1px solid #eee',
          display: 'flex',
          gap: 1,
        }}
      >
        <TextField
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={
            conditionId
              ? 'Write a message'
              : 'Select a condition to send messages'
          }
          fullWidth
          size="small"
          disabled={loading || !conditionId}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !loading) send();
          }}
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
          {loading ? (
            <CircularProgress size={24} color="inherit" />
          ) : (
            <SendIcon />
          )}
        </IconButton>
      </Box>
    </Card>
  );
}
