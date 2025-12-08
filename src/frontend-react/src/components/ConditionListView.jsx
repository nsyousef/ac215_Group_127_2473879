'use client';

import { useState } from 'react';
import { Box, List, Button, Avatar, Typography, ListItemButton, ListItemText, useTheme, Card, CardContent, Checkbox, Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions } from '@mui/material';
import { AddOutlined, DeleteOutlined, CancelOutlined } from '@mui/icons-material';
import { useDiseaseContext } from '@/contexts/DiseaseContext';
import { isElectron } from '@/utils/config';

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
    const { diseases, reload: reloadDiseases } = useDiseaseContext();
    const rawConditions = propConditions || diseases || [];
    
    const [deleteMode, setDeleteMode] = useState(false);
    const [selectedForDelete, setSelectedForDelete] = useState(new Set());
    const [showDeleteDialog, setShowDeleteDialog] = useState(false);
    const [isDeleting, setIsDeleting] = useState(false);

    const theme = useTheme();
    const primary = theme.palette.primary.main;

    // Deduplicate conditions by ID (keep the last occurrence which has the most recent data)
    const deduplicatedConditions = rawConditions.reduce((acc, condition) => {
        acc[condition.id] = condition; // Later occurrences override earlier ones
        return acc;
    }, {});
    
    // Convert back to array
    const uniqueConditions = Object.values(deduplicatedConditions);

    // Sort conditions by last message timestamp (most recent first)
    const conditions = [...uniqueConditions].sort((a, b) => {
        const timestampA = a.lastMessageTimestamp || a.date || '';
        const timestampB = b.lastMessageTimestamp || b.date || '';
        
        // If no timestamp, put at the end
        if (!timestampA && !timestampB) return 0;
        if (!timestampA) return 1;
        if (!timestampB) return -1;
        
        // Compare timestamps (most recent first)
        return new Date(timestampB).getTime() - new Date(timestampA).getTime();
    });

    // Helper function to format date and create display name
    const formatDisplayName = (condition) => {
        if (!condition.date) {
            return condition.name || 'Unknown Condition';
        }
        
        // Extract date part (remove timestamp if present, e.g., "2025-11-21_123456" -> "2025-11-21")
        let dateStr = condition.date;
        if (dateStr.includes('_')) {
            dateStr = dateStr.split('_')[0];
        }
        
        // Format date nicely (e.g., "2025-11-21" -> "Nov 21, 2025")
        try {
            const date = new Date(dateStr + 'T00:00:00');
            const formattedDate = date.toLocaleDateString('en-US', { 
                year: 'numeric', 
                month: 'short', 
                day: 'numeric' 
            });
            return `${formattedDate} - ${condition.name || 'Unknown Condition'}`;
        } catch (e) {
            // Fallback to original format if date parsing fails
            return `${dateStr} - ${condition.name || 'Unknown Condition'}`;
        }
    };

    const handleToggleDeleteMode = () => {
        setDeleteMode(!deleteMode);
        setSelectedForDelete(new Set());
    };

    const handleToggleSelect = (conditionId) => {
        setSelectedForDelete(prev => {
            const next = new Set(prev);
            if (next.has(conditionId)) {
                next.delete(conditionId);
            } else {
                next.add(conditionId);
            }
            return next;
        });
    };

    const handleDeleteClick = () => {
        if (selectedForDelete.size > 0) {
            setShowDeleteDialog(true);
        }
    };

    const handleConfirmDelete = async () => {
        if (selectedForDelete.size === 0) return;
        
        setIsDeleting(true);
        try {
            const caseIds = Array.from(selectedForDelete);
            
            // Check if currently selected condition is being deleted
            const isSelectedConditionDeleted = selectedConditionId && caseIds.includes(selectedConditionId);
            
            if (isElectron() && window.electronAPI?.deleteCases) {
                await window.electronAPI.deleteCases(caseIds);
            } else {
                console.warn('Delete cases not available in non-Electron environment');
            }
            
            // Reload diseases list
            await reloadDiseases();
            
            // If the selected condition was deleted, clear the selection
            if (isSelectedConditionDeleted && onChange) {
                onChange(null);
            }
            
            // Clear selection and exit delete mode
            setSelectedForDelete(new Set());
            setDeleteMode(false);
            setShowDeleteDialog(false);
        } catch (error) {
            console.error('Failed to delete cases:', error);
            alert('Failed to delete cases. Please try again.');
        } finally {
            setIsDeleting(false);
        }
    };

    return (
        <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Delete mode header */}
            {deleteMode && (
                <Box sx={{ borderBottom: '1px solid', borderColor: 'divider', p: 1.5, display: 'flex', alignItems: 'center', justifyContent: 'space-between', bgcolor: 'error.light', color: 'error.contrastText' }}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {selectedForDelete.size > 0 
                            ? `${selectedForDelete.size} case${selectedForDelete.size > 1 ? 's' : ''} selected`
                            : 'Select cases to delete'}
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                        {selectedForDelete.size > 0 && (
                            <Button
                                variant="contained"
                                color="error"
                                size="small"
                                startIcon={<DeleteOutlined />}
                                onClick={handleDeleteClick}
                                sx={{ textTransform: 'none' }}
                            >
                                Delete
                            </Button>
                        )}
                        <Button
                            variant="outlined"
                            size="small"
                            startIcon={<CancelOutlined />}
                            onClick={handleToggleDeleteMode}
                            sx={{ textTransform: 'none', bgcolor: 'background.paper' }}
                        >
                            Cancel
                        </Button>
                    </Box>
                </Box>
            )}

            {/* Delete mode toggle button (when not in delete mode) */}
            {!deleteMode && (
                <Box sx={{ borderBottom: '1px solid', borderColor: 'divider', p: 1.5 }}>
                    <Button
                        variant="outlined"
                        color="error"
                        size="small"
                        startIcon={<DeleteOutlined />}
                        onClick={handleToggleDeleteMode}
                        fullWidth
                        sx={{ textTransform: 'none' }}
                    >
                        Delete Cases
                    </Button>
                </Box>
            )}

            {/* Scrollable list area */}
            <CardContent sx={{ flex: 1, overflow: 'auto', p: 0 }}>
                <List sx={{ bgcolor: 'background.paper' }} disablePadding>
                    {conditions.map((condition) => (
                        <div key={condition.id}>
                            <ListItemButton
                                selected={!deleteMode && selectedConditionId === condition.id}
                                onClick={() => {
                                    if (deleteMode) {
                                        handleToggleSelect(condition.id);
                                    } else {
                                        onChange(condition);
                                    }
                                }}
                                sx={{
                                    px: compact ? 1 : 2,
                                    py: compact ? 0.75 : 1.25,
                                    borderTop: '1px solid',
                                    borderColor: 'divider',
                                    bgcolor: deleteMode && selectedForDelete.has(condition.id) ? 'error.light' : 'transparent',
                                    '&.Mui-selected': {
                                        bgcolor: `${primary}20`,
                                        borderLeft: `4px solid ${primary}`,
                                    },
                                    '&:hover': {
                                        bgcolor: deleteMode && selectedForDelete.has(condition.id) 
                                            ? 'error.main' 
                                            : `${primary}10`,
                                    },
                                }}
                            >
                                {deleteMode && (
                                    <Checkbox
                                        checked={selectedForDelete.has(condition.id)}
                                        onChange={() => handleToggleSelect(condition.id)}
                                        onClick={(e) => e.stopPropagation()}
                                        sx={{ mr: 1 }}
                                    />
                                )}
                                <ListItemText
                                    primary={<Typography variant="subtitle1" sx={{ fontWeight: 600, fontSize: '1rem' }}>{formatDisplayName(condition)}</Typography>}
                                />
                                {/* Show image thumbnail if available, otherwise show avatar */}
                                {condition.image ? (
                                    <Box
                                        sx={{
                                            width: compact ? 28 : 40,
                                            height: compact ? 28 : 40,
                                            borderRadius: '4px',
                                            overflow: 'hidden',
                                            ml: 1,
                                            flexShrink: 0,
                                        }}
                                    >
                                        <img
                                            src={condition.image}
                                            alt={condition.name}
                                            style={{
                                                width: '100%',
                                                height: '100%',
                                                objectFit: 'cover',
                                            }}
                                        />
                                    </Box>
                                ) : (
                                    <Avatar sx={{ bgcolor: 'grey.200', width: compact ? 28 : 32, height: compact ? 28 : 32, ml: 1 }} />
                                )}
                            </ListItemButton>
                        </div>
                    ))}
                </List>
            </CardContent>

            {/* Persistent Add Disease button at bottom */}
            {showAddButton && onAddDisease && !deleteMode && (
                <Box sx={{ borderTop: '1px solid', borderColor: 'divider', p: 2, backgroundColor: 'background.paper' }}>
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

            {/* Delete confirmation dialog */}
            <Dialog
                open={showDeleteDialog}
                onClose={() => !isDeleting && setShowDeleteDialog(false)}
            >
                <DialogTitle>Delete Cases</DialogTitle>
                <DialogContent>
                    <DialogContentText>
                        Are you sure you want to delete {selectedForDelete.size} case{selectedForDelete.size > 1 ? 's' : ''}? 
                        This action cannot be undone and will permanently remove all data for these cases.
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button 
                        onClick={() => setShowDeleteDialog(false)} 
                        disabled={isDeleting}
                        sx={{ textTransform: 'none' }}
                    >
                        Cancel
                    </Button>
                    <Button 
                        onClick={handleConfirmDelete} 
                        color="error" 
                        variant="contained"
                        disabled={isDeleting}
                        sx={{ textTransform: 'none' }}
                    >
                        {isDeleting ? 'Deleting...' : 'Delete'}
                    </Button>
                </DialogActions>
            </Dialog>
        </Card>
    );
}
