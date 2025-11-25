'use client';

import { useState } from 'react';
import Image from 'next/image';
import { Box, useTheme } from '@mui/material';
import { useDiseaseContext } from '@/contexts/DiseaseContext';
import ConditionPopover from './ConditionPopover';

/**
 * Reusable body map component with interactive spots
 * @param {Object} props
 * @param {number} props.selectedConditionId - Currently selected condition ID
 * @param {Function} props.onSpotClick - Callback when a spot is clicked: (conditionId) => void
 * @param {string} props.maxWidth - Max width of the body map container (default: '280px')
 * @param {boolean} props.showPopover - If true, show popover on mobile; if false, direct to results (default: false)
 * @param {Function} props.onPopoverViewResults - Callback when popover's "View Results" is clicked (mobile only)
 */
export default function BodyMapView({
    selectedConditionId,
    onSpotClick,
    maxWidth = '280px',
    showPopover = false,
    onPopoverViewResults,
    conditions: propConditions,
}) {
    const [popoverCondition, setPopoverCondition] = useState(null);
    const { diseases } = useDiseaseContext();
    const conditions = propConditions || diseases || [];

    const handleSpotClick = (conditionId) => {
        const condition = (conditions || []).find((c) => c.id === conditionId);

        if (showPopover && condition) {
            // On mobile: show popover AND select the condition
            setPopoverCondition(condition);
            // Call the callback to select the condition (parent will update selectedConditionId)
            onSpotClick(conditionId);
        } else {
            // On desktop: direct callback
            onSpotClick(conditionId);
        }
    };

    const handlePopoverViewResults = () => {
        if (onPopoverViewResults && popoverCondition) {
            onPopoverViewResults(popoverCondition);
            setPopoverCondition(null);
        }
    };

    const handlePopoverDismiss = () => {
        setPopoverCondition(null);
    };
    const theme = useTheme();
    const primary = theme.palette.primary.main;

    return (
        <>
            <Box
                sx={{
                    position: 'relative',
                    width: '100%',
                    maxWidth,
                    aspectRatio: '1 / 1.3',
                }}
            >
                <Image
                    src="/assets/human_body.png"
                    alt="Body Map"
                    fill
                    style={{ objectFit: 'contain' }}
                    priority
                />

                {/* Interactive spots overlay: use saved mapPosition */}
                {conditions.map((condition) => {
                    // Use condition.mapPosition (normalized percents) for spot position
                    let top = null;
                    let left = null;
                    if (condition.mapPosition && typeof condition.mapPosition.leftPct === 'number' && typeof condition.mapPosition.topPct === 'number') {
                        left = `${condition.mapPosition.leftPct}%`;
                        top = `${condition.mapPosition.topPct}%`;
                    }

                    if (left == null || top == null) return null; // no known position

                    const isSelected = selectedConditionId === condition.id;

                    return (
                        <Box
                            key={condition.id}
                            sx={{
                                position: 'absolute',
                                top,
                                left,
                                transform: 'translate(-50%, -50%)',
                                width: isSelected ? 18 : 14,
                                height: isSelected ? 18 : 14,
                                bgcolor: '#e74c3c',
                                borderRadius: '50%',
                                border: '2px solid #fff',
                                cursor: 'pointer',
                                boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
                                transition: 'all 0.2s',
                                outline: isSelected ? `2px solid ${primary}` : 'none',
                                outlineOffset: '3px',
                                '&:hover': {
                                    width: 18,
                                    height: 18,
                                    boxShadow: '0 4px 8px rgba(0,0,0,0.4)',
                                },
                            }}
                            onClick={() => handleSpotClick(condition.id)}
                            title={condition.name}
                        />
                    );
                })}
            </Box>

            {/* Mobile popover: only show when showPopover=true */}
            {showPopover && popoverCondition && (
                <ConditionPopover
                    condition={popoverCondition}
                    onViewResults={handlePopoverViewResults}
                    onDismiss={handlePopoverDismiss}
                />
            )}
        </>
    );
}
