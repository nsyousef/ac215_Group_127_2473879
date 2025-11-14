'use client';

import Image from 'next/image';
import { Box, Card, CardContent, Typography, Button, ButtonGroup, useTheme } from '@mui/material';
import { AssignmentReturnOutlined, QuestionAnswerOutlined } from '@mui/icons-material';

/**
 * Reusable results/overview panel component
 * @param {Object} props
 * @param {Object} props.selectedCondition - Selected condition object { id, name, description }
 * @param {boolean} props.showActions - If true, show Track Progress and Ask Question buttons (default: true)
 */
export default function ResultsPanel({ selectedCondition, showActions = true, isMobile = false, onBack, onTrack, onAsk, onCombined }) {
    const theme = useTheme();
    const primary = theme.palette.primary.main;
    const primaryHover = '#067891';

    return (
        <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <CardContent sx={{ flex: 1, overflow: 'auto' }}>
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
                            <Image
                                src="/nasty_skin.jpg"
                                alt={selectedCondition.name}
                                fill
                                style={{ objectFit: 'cover' }}
                            />
                        </Box>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                            {selectedCondition.name}
                        </Typography>
                        <Typography variant="body2" sx={{ color: '#666', mb: 2 }}>
                            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
                        </Typography>
                        <Typography variant="body2" sx={{ color: '#666', mb: 3 }}>
                            Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
                        </Typography>
                    </>
                ) : (
                    <Typography variant="body2" sx={{ color: '#999', textAlign: 'center', py: 4 }}>
                        Select a condition to view details
                    </Typography>
                )}
            </CardContent>
            {/* Action buttons - only shown when a condition is selected */}
            {showActions && selectedCondition && (
                <Box sx={{ p: 2, borderTop: '1px solid #e0e0e0' }}>
                    {/* If onCombined is provided, show a single combined button (desktop case) */}
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
                            <Button startIcon={<AssignmentReturnOutlined />} sx={{ bgcolor: primary, '&:hover': { bgcolor: primaryHover } }} onClick={() => onTrack && onTrack(selectedCondition)}>
                                Track Progress
                            </Button>
                            <Button startIcon={<QuestionAnswerOutlined />} sx={{ bgcolor: primary, '&:hover': { bgcolor: primaryHover } }} onClick={() => onAsk && onAsk(selectedCondition)}>
                                Ask Question
                            </Button>
                        </ButtonGroup>
                    )}
                </Box>
            )}
        </Card>
    );
}
