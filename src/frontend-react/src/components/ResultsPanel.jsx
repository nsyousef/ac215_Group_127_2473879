'use client';

import { useEffect, useRef, useState } from 'react';
import Image from 'next/image';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  ButtonGroup,
  useTheme,
} from '@mui/material';
import {
  AssignmentReturnOutlined,
  QuestionAnswerOutlined,
} from '@mui/icons-material';
import { isElectron } from '@/utils/config';

/**
 * Reusable results/overview panel component
 * @param {Object} props
 * @param {Object} props.selectedCondition - Selected condition object { id, name, description, llmResponse, image, caseId? }
 * @param {boolean} props.showActions - If true, show Track Progress and Ask Question buttons (default: true)
 */
export default function ResultsPanel({
  selectedCondition,
  showActions = true,
  isMobile = false,
  onBack,
  onTrack,
  onAsk,
  onCombined,
}) {
  const theme = useTheme();
  const primary = theme.palette.primary.main;
  const primaryHover = '#067891';

  // Local streaming text state
  const [explanation, setExplanation] = useState(
    selectedCondition
      ? selectedCondition.llmResponse ||
          selectedCondition.description ||
          ''
      : ''
  );

  // Ref to auto-scroll the content
  const contentRef = useRef(null);

  // Whenever the selected condition changes or its final llmResponse updates,
  // sync the local explanation.
  useEffect(() => {
    if (!selectedCondition) {
      setExplanation('');
      return;
    }
    setExplanation(
      selectedCondition.llmResponse ||
        selectedCondition.description ||
        ''
    );
  }, [
    selectedCondition?.id,
    selectedCondition?.llmResponse,
    selectedCondition?.description,
  ]);

  // Subscribe to initial LLM explanation streaming from Electron.
  // This uses the generic mlOnStreamChunk hook that main.js emits during
  // the 'predict' (initial analysis) pipeline.
  useEffect(() => {
    if (!selectedCondition) return;
    if (!isElectron()) return;
    if (typeof window === 'undefined') return;
    if (!window.electronAPI?.mlOnStreamChunk) return;

    // Handler receives each streamed text chunk
    const unsubscribe = window.electronAPI.mlOnStreamChunk((chunk) => {
      if (!chunk) return;
      setExplanation((prev) => (prev || '') + chunk);
    });

    return () => {
      if (typeof unsubscribe === 'function') unsubscribe();
    };
  }, [selectedCondition?.id]);

  // Auto-scroll the CardContent as explanation grows
  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [explanation]);

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <CardContent
        ref={contentRef}
        sx={{ flex: 1, overflow: 'auto' }}
      >
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
          {selectedCondition ? 'Results' : 'Overview'}
        </Typography>

        {selectedCondition ? (
          <>
            <Box
              sx={{
                position: 'relative',
                width: '100%',
                height: 200,
                bgcolor: '#e0e0e0',
                borderRadius: 1,
                mb: 2,
                overflow: 'hidden',
              }}
            >
              {selectedCondition.image ? (
                <img
                  src={selectedCondition.image}
                  alt={selectedCondition.name}
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                  }}
                />
              ) : (
                <Image
                  src="/nasty_skin.jpg"
                  alt={selectedCondition.name}
                  fill
                  style={{ objectFit: 'cover' }}
                />
              )}
            </Box>

            <Typography
              variant="subtitle2"
              sx={{ fontWeight: 600, mb: 1 }}
            >
              {selectedCondition.name}
            </Typography>

            <Typography
              variant="body2"
              sx={{
                color: '#666',
                mb: 2,
                whiteSpace: 'pre-wrap',
              }}
            >
              {explanation ||
                'Preparing your explanation…'}
            </Typography>
          </>
        ) : (
          <Typography
            variant="body2"
            sx={{ color: '#999', textAlign: 'center', py: 4 }}
          >
            Select a condition to view details
          </Typography>
        )}
      </CardContent>

      {showActions && selectedCondition && (
        <Box sx={{ p: 2, borderTop: '1px solid #e0e0e0' }}>
          {onCombined ? (
            <Button
              variant="contained"
              fullWidth
              startIcon={<QuestionAnswerOutlined />}
              onClick={() => onCombined && onCombined(selectedCondition)}
              sx={{ textTransform: 'none', py: 1 }}
            >
              Chat & Track
            </Button>
          ) : (
            <ButtonGroup
              variant="contained"
              fullWidth
              sx={{
                '& > button': {
                  flex: 1,
                  py: 1,
                  textTransform: 'none',
                  fontSize: '0.9rem',
                },
              }}
            >
              <Button
                startIcon={<AssignmentReturnOutlined />}
                sx={{ bgcolor: primary, '&:hover': { bgcolor: primaryHover } }}
                onClick={() => onTrack && onTrack(selectedCondition)}
              >
                Track Progress
              </Button>
              <Button
                startIcon={<QuestionAnswerOutlined />}
                sx={{ bgcolor: primary, '&:hover': { bgcolor: primaryHover } }}
                onClick={() => onAsk && onAsk(selectedCondition)}
              >
                Ask Question
              </Button>
            </ButtonGroup>
          )}
        </Box>
      )}
    </Card>
  );
}
