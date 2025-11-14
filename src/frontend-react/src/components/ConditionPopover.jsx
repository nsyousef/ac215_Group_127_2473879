'use client';

import { Box, Card, CardContent, Typography, Button, IconButton, useTheme } from '@mui/material';
import { ArrowForwardOutlined, CloseOutlined } from '@mui/icons-material';

/**
 * Mini popover/card that appears when tapping a body map spot on mobile
 * @param {Object} props
 * @param {Object} props.condition - Condition object { id, name, description }
 * @param {Function} props.onViewResults - Callback when "View Results" is clicked
 * @param {Function} props.onDismiss - Callback when popover is dismissed (close button or outside click)
 */
export default function ConditionPopover({ condition, onViewResults, onDismiss }) {
    if (!condition) return null;

    const handleBackdropClick = () => {
        if (onDismiss) {
            onDismiss();
        }
    };

    return (
        <>
            {/* Semi-transparent backdrop that dismisses the popover when clicked */}
            <Box
                onClick={handleBackdropClick}
                sx={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    zIndex: 999,
                    // Transparent - just for click detection
                }}
            />

            {/* Popover Card */}
            <Card
                sx={{
                    position: 'fixed',
                    bottom: '80px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    width: 'calc(100% - 32px)',
                    maxWidth: '300px',
                    zIndex: 1000,
                    boxShadow: '0 8px 24px rgba(0,0,0,0.15)',
                    animation: 'slideUp 0.3s ease-out',
                    '@keyframes slideUp': {
                        from: {
                            opacity: 0,
                            transform: 'translateX(-50%) translateY(20px)',
                        },
                        to: {
                            opacity: 1,
                            transform: 'translateX(-50%) translateY(0)',
                        },
                    },
                }}
            >
                <CardContent sx={{ pb: 1, position: 'relative' }}>
                    {/* Close button */}
                    <IconButton
                        size="small"
                        onClick={handleBackdropClick}
                        sx={{
                            position: 'absolute',
                            top: 4,
                            right: 4,
                            color: '#999',
                            '&:hover': {
                                color: '#666',
                            },
                        }}
                    >
                        <CloseOutlined fontSize="small" />
                    </IconButton>

                    <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1, pr: 3 }}>
                        {condition.name}
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#666', display: 'block', mb: 2 }}>
                        {condition.description}
                    </Typography>
                    <Button
                        variant="contained"
                        size="small"
                        endIcon={<ArrowForwardOutlined />}
                        fullWidth
                        onClick={onViewResults}
                        sx={{
                            textTransform: 'none',
                            bgcolor: useTheme().palette.primary.main,
                            '&:hover': {
                                bgcolor: '#067891',
                            },
                        }}
                    >
                        View Results
                    </Button>
                </CardContent>
            </Card>
        </>
    );
}
