'use client';

import { Box, Card, CardContent, Avatar, Typography, List, Button } from '@mui/material';
import { AddOutlined } from '@mui/icons-material';
import { useDiseaseContext } from '@/contexts/DiseaseContext';

/**
 * Reusable condition list component
 * @param {Object} props
 * @param {number} props.selectedConditionId - Currently selected condition ID
 * @param {Function} props.onChange - Callback when a condition is selected: (condition) => void
 * @param {boolean} props.compact - If true, reduces padding for desktop view (default: false)
 * @param {Function} props.onAddDisease - Callback when "Add Disease" button is clicked
 */
export default function ConditionListView({ selectedConditionId, onChange, compact = false, conditions: propConditions, onAddDisease }) {
    const { diseases } = useDiseaseContext();
    const conditions = propConditions || diseases || [];

    return (
        <Box>
            <List sx={{ bgcolor: '#fff', borderRadius: 1 }}>
                {conditions.map((condition) => (
                    <Card
                        key={condition.id}
                        sx={{
                            mb: compact ? 1 : 1.5,
                            cursor: 'pointer',
                            bgcolor: selectedConditionId === condition.id ? '#e3f2fd' : '#fff',
                            border: selectedConditionId === condition.id ? '2px solid #1976d2' : '1px solid #e0e0e0',
                            transition: 'all 0.2s',
                            '&:hover': {
                                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                            },
                        }}
                        onClick={() => onChange(condition)}
                    >
                        <CardContent sx={{ py: compact ? 1.5 : 2, px: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Box sx={{ flex: 1 }}>
                                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                                    {condition.name}
                                </Typography>
                                <Typography variant="caption" sx={{ color: '#999' }}>
                                    {condition.description}
                                </Typography>
                            </Box>
                            <Avatar sx={{ bgcolor: '#e0e0e0', width: compact ? 28 : 32, height: compact ? 28 : 32, ml: 1 }} />
                        </CardContent>
                    </Card>
                ))}
            </List>

            {/* Add Disease button at bottom (desktop only) */}
            {onAddDisease && (
                <Button
                    variant="contained"
                    startIcon={<AddOutlined />}
                    onClick={onAddDisease}
                    fullWidth
                    sx={{ 
                        mt: 2, 
                        textTransform: 'none', 
                        py: 1.2,
                        display: { xs: 'none', md: 'block' }
                    }}
                >
                    Add Disease
                </Button>
            )}
        </Box>
    );
}
