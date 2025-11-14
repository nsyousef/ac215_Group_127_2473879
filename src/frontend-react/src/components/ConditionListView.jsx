'use client';

import { Box, List, Button, Avatar, Typography, ListItemButton, ListItemText, useTheme } from '@mui/material';
import { AddOutlined } from '@mui/icons-material';
import { useDiseaseContext } from '@/contexts/DiseaseContext';

/**
 * Reusable condition list component
 * @param {Object} props
 * @param {number} props.selectedConditionId - Currently selected condition ID
 * @param {Function} props.onChange - Callback when a condition is selected: (condition) => void
 * @param {boolean} props.compact - If true, reduces padding for desktop view (default: false)
 * @param {Function} props.onAddDisease - Callback when "Add Disease" button is clicked
 * @param {boolean} props.showAddButton - If false, hide the internal Add Disease button (desktop persistent footer used instead)
 */
export default function ConditionListView({ selectedConditionId, onChange, compact = false, conditions: propConditions, onAddDisease, showAddButton = true }) {
    const { diseases } = useDiseaseContext();
    const conditions = propConditions || diseases || [];

    const theme = useTheme();
    const primary = theme.palette.primary.main;

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Scrollable list area */}
            <Box sx={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
                <List sx={{ bgcolor: 'background.paper', borderRadius: 1 }} disablePadding>
                    {conditions.map((condition) => (
                        <div key={condition.id}>
                            <ListItemButton
                                selected={selectedConditionId === condition.id}
                                onClick={() => onChange(condition)}
                                sx={{
                                    px: compact ? 1 : 2,
                                    py: compact ? 0.75 : 1.25,
                                    borderTop: '1px solid',
                                    borderColor: 'divider',
                                    '&.Mui-selected': {
                                        bgcolor: `${primary}20`,
                                        borderLeft: `4px solid ${primary}`,
                                    },
                                }}
                            >
                                <ListItemText
                                    primary={<Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{condition.name}</Typography>}
                                    secondary={<Typography variant="caption" sx={{ color: 'text.secondary' }}>{condition.description}</Typography>}
                                />
                                <Avatar sx={{ bgcolor: 'grey.200', width: compact ? 28 : 32, height: compact ? 28 : 32, ml: 1 }} />
                            </ListItemButton>
                        </div>
                    ))}
                </List>
            </Box>

            {/* Persistent Add Disease button at bottom */}
            {showAddButton && onAddDisease && (
                <Box sx={{ borderTop: '1px solid', borderColor: 'divider', pt: 2, backgroundColor: 'background.paper' }}>
                    <Button
                        variant="contained"
                        color="primary"
                        startIcon={<AddOutlined />}
                        onClick={onAddDisease}
                        fullWidth
                        size="large"
                        sx={{
                            textTransform: 'none',
                        }}
                        aria-label="Add disease"
                    >
                        Add Disease
                    </Button>
                </Box>
            )}
        </Box>
    );
}
