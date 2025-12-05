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

  // Only show predictionText - never show llmResponse or streaming
  const [explanation, setExplanation] = useState(() => {
    if (!selectedCondition) return '';
    // Only use predictionText - never show llmResponse
    if (selectedCondition.predictionText) {
      hasPredictionTextRef.current = true;
      return selectedCondition.predictionText;
    }
    // If no predictionText, show empty or description (but never llmResponse)
    return selectedCondition.description || '';
  });

  // Ref to auto-scroll the content
  const contentRef = useRef(null);
  const prevExplanationLengthRef = useRef(0);
  const isStreamingRef = useRef(false);
  const hasPredictionTextRef = useRef(false); // Track if we've received predictionText via event

  // Scroll to top when a new condition is selected
  useEffect(() => {
    if (contentRef.current && selectedCondition) {
      contentRef.current.scrollTop = 0;
      prevExplanationLengthRef.current = 0;
      isStreamingRef.current = false;
      hasPredictionTextRef.current = false; // Reset when condition changes
    }
  }, [selectedCondition?.id]);

  // Whenever the selected condition changes or its predictionText/llmResponse updates,
  // sync the local explanation. Always prioritize predictionText.
  useEffect(() => {
    if (!selectedCondition) {
      setExplanation('');
      prevExplanationLengthRef.current = 0;
      isStreamingRef.current = false;
      hasPredictionTextRef.current = false;
      return;
    }
    // Always prioritize predictionText if it exists - set it immediately
    if (selectedCondition.predictionText) {
      hasPredictionTextRef.current = true; // Mark that we have predictionText
      setExplanation(selectedCondition.predictionText);
      prevExplanationLengthRef.current = selectedCondition.predictionText.length;
      isStreamingRef.current = false;
      return;
    }
    // Only fall back to description if we don't have predictionText
    // NEVER use llmResponse - we only show predictionText
    if (!hasPredictionTextRef.current) {
      const newExplanation = selectedCondition.description || '';
      setExplanation(newExplanation);
      prevExplanationLengthRef.current = newExplanation.length;
    }
    isStreamingRef.current = false;
  }, [
    selectedCondition?.id,
    selectedCondition?.predictionText,
    selectedCondition?.description,
  ]);

  // Subscribe to predictionText metadata (sent before LLM streaming starts)
  // This MUST run before the streaming subscription
  useEffect(() => {
    if (!selectedCondition) {
      hasPredictionTextRef.current = false;
      return;
    }
    
    // If condition already has predictionText, set the flag immediately
    if (selectedCondition.predictionText) {
      hasPredictionTextRef.current = true;
      return;
    }
    
    if (!isElectron()) return;
    if (typeof window === 'undefined') return;
    if (!window.electronAPI?.mlOnPredictionText) return;

    // Reset flag when condition changes (only if no predictionText on condition)
    hasPredictionTextRef.current = false;

    // Handler receives predictionText as soon as ML inference is done
    const unsubscribe = window.electronAPI.mlOnPredictionText((predictionText) => {
      if (!predictionText) return;
      // Set predictionText immediately - this happens before LLM streaming
      hasPredictionTextRef.current = true;
      setExplanation(predictionText);
      isStreamingRef.current = false;
      prevExplanationLengthRef.current = predictionText.length;
      
    });

    return () => {
      if (typeof unsubscribe === 'function') unsubscribe();
    };
  }, [selectedCondition?.id]);

  // REMOVED: LLM streaming subscription - we only use predictionText, no streaming

  // Auto-scroll removed - we don't stream anymore, only show predictionText

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
