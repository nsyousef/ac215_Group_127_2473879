'use client';

import { useState, useEffect } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import {
    Box,
    Container,
    Card,
    Grid,
    useMediaQuery,
    useTheme,
    Typography,
    Button,
} from '@mui/material';
import { ArrowBackOutlined } from '@mui/icons-material';
import BodyMapView from '@/components/BodyMapView';
import ConditionListView from '@/components/ConditionListView';
import ResultsPanel from '@/components/ResultsPanel';
import TimeTrackingPanel from '@/components/TimeTrackingPanel';
import ChatPanel from '@/components/ChatPanel';
import MobileLayout from '@/components/layouts/MobileLayout';
import AddDiseaseFlow from '@/components/AddDiseaseFlow';
import { useDiseaseContext } from '@/contexts/DiseaseContext';

export default function Home() {
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('md'));
    const router = useRouter();
    const searchParams = useSearchParams();

    // Get view from URL query params (for mobile navigation)
    const viewParam = searchParams.get('view') || 'home';

    const [selectedCondition, setSelectedCondition] = useState(null);
    const [mobileView, setMobileView] = useState(viewParam);
    const [previousMobileView, setPreviousMobileView] = useState('home'); // Track previous view for back navigation

    const { diseases } = useDiseaseContext();

    // Sync mobileView with URL query param and track previous view
    useEffect(() => {
        // Before updating mobileView, save the current view as previous (for detail screens)
        if (['home', 'map'].includes(mobileView)) {
            setPreviousMobileView(mobileView);
        }
        setMobileView(viewParam);
    }, [viewParam]);

    const handleSelectCondition = (condition) => {
        setSelectedCondition(condition);
        // On mobile, when selecting from list, show results
        if (isMobile) {
            router.push('/?view=results');
        }
    };

    const handleSpotClick = (conditionId) => {
        // Called when a spot is clicked on the body map
        // Works on both mobile (with popover) and desktop
        const condition = (diseases || []).find((c) => c.id === conditionId);
        if (condition) {
            setSelectedCondition(condition);
        }
    };

    const handlePopoverViewResults = (condition) => {
        // When user clicks "View Results" in the popover (mobile only)
        setSelectedCondition(condition);
        router.push('/?view=results');
    };

    const handleBackFromResults = () => {
        // Go back to the previous view
        // If on results, go back to the previous home/map view
        // If on chat/time, go back to results
        if (mobileView === 'results') {
            router.push(`/?view=${previousMobileView}`);
        } else if (['chat', 'time'].includes(mobileView)) {
            router.push('/?view=results');
        }
    };

    const handleOpenTime = (condition) => {
        if (!condition) return;
        router.push('/?view=time');
    };

    const handleOpenChat = (condition) => {
        if (!condition) return;
        router.push('/?view=chat');
    };

    const handleAddDisease = () => {
        setShowAddFlow(true);
    };

    // Add disease modal state
    const [showAddFlow, setShowAddFlow] = useState(false);

    const handleAddSaved = (newDisease) => {
        // Close modal and select condition, navigate to results/detail
        setShowAddFlow(false);
        setSelectedCondition(newDisease);
        router.push('/?view=results');
    };

    // ========== MOBILE LAYOUT ==========
    if (isMobile) {
        // Show bottom nav only on home/map screens; show back button on detail screens
        const showBottomNav = ['home', 'map'].includes(mobileView);
        const showBackButton = ['results', 'chat', 'time'].includes(mobileView);

        return (
            <MobileLayout
                currentPage={mobileView === 'home' ? 'list' : mobileView === 'map' ? 'body-map' : 'results'}
                showBottomNav={showBottomNav}
                showBackButton={showBackButton}
                onBack={handleBackFromResults}
                onAddDisease={handleAddDisease}
            >
                <Container maxWidth="sm" sx={{ py: 2, pb: showBottomNav ? 9 : 2 }}>
                    {/* Home/List View */}
                    {mobileView === 'home' && (
                        <>
                            <Typography variant="body2" sx={{ color: '#666', textAlign: 'center', mb: 2 }}>
                                Select a condition to view details.
                            </Typography>
                            <ConditionListView
                                selectedConditionId={selectedCondition?.id}
                                onChange={handleSelectCondition}
                                onAddDisease={handleAddDisease}
                            />
                        </>
                    )}

                    {/* Body Map View */}
                    {mobileView === 'map' && (
                        <>
                            <Typography variant="body2" sx={{ color: '#666', textAlign: 'center', mb: 2 }}>
                                Tap a spot on the body map.
                            </Typography>
                            <Card sx={{ p: 2, mb: 2, display: 'flex', justifyContent: 'center' }}>
                                <BodyMapView
                                    selectedConditionId={selectedCondition?.id}
                                    onSpotClick={handleSpotClick}
                                    maxWidth="260px"
                                    showPopover={true}
                                    onPopoverViewResults={handlePopoverViewResults}
                                />
                            </Card>
                        </>
                    )}

                    {/* Results View */}
                    {mobileView === 'results' && (
                        <ResultsPanel
                            selectedCondition={selectedCondition}
                            isMobile={true}
                            onBack={handleBackFromResults}
                            onTrack={handleOpenTime}
                            onAsk={handleOpenChat}
                        />
                    )}

                    {/* Time Tracking View */}
                    {mobileView === 'time' && (
                        <>
                            <TimeTrackingPanel conditionId={selectedCondition?.id} />
                        </>
                    )}

                    {/* Chat View */}
                    {mobileView === 'chat' && (
                        <>
                            <ChatPanel conditionId={selectedCondition?.id} />
                        </>
                    )}

                    {/* Show empty state if no condition selected and not in results view */}
                    {!selectedCondition && mobileView !== 'results' && (
                        <Typography variant="body2" sx={{ color: '#999', textAlign: 'center', py: 4 }}>
                            No condition selected
                        </Typography>
                    )}
                </Container>
                {/* Add disease modal (shared) */}
                <AddDiseaseFlow open={showAddFlow} onClose={() => setShowAddFlow(false)} onSaved={handleAddSaved} />
            </MobileLayout>
        );
    }

    // ========== DESKTOP LAYOUT ==========
    // If the desktop view param is 'home' show the home/list+map layout.
    if (viewParam === 'home') {
        return (
            <Box sx={{ minHeight: '100vh', bgcolor: '#f5f5f5', py: 3 }}>
                <Container maxWidth="xl" sx={{ mb: 3 }}>
                    <Typography variant="h5" sx={{ fontWeight: 600, mb: 3 }}>
                        Home
                    </Typography>

                    <Grid container spacing={3}>
                        {/* Left: Conditions list */}
                        <Grid item xs={12} md={4}>
                            <ConditionListView
                                selectedConditionId={selectedCondition?.id}
                                onChange={handleSelectCondition}
                                onAddDisease={handleAddDisease}
                            />
                        </Grid>

                        {/* Center: Body map */}
                        <Grid item xs={12} md={4}>
                            <Card sx={{ p: 2, mb: 2, display: 'flex', justifyContent: 'center' }}>
                                <BodyMapView
                                    selectedConditionId={selectedCondition?.id}
                                    onSpotClick={handleSpotClick}
                                    maxWidth="420px"
                                    showPopover={false}
                                />
                            </Card>
                        </Grid>

                        {/* Right: Results with single combined button to open full page */}
                        <Grid item xs={12} md={4}>
                            <ResultsPanel
                                selectedCondition={selectedCondition}
                                showActions={true}
                                onCombined={() => router.push('/?view=results')}
                            />
                        </Grid>
                    </Grid>
                </Container>
                <AddDiseaseFlow open={showAddFlow} onClose={() => setShowAddFlow(false)} onSaved={handleAddSaved} />
            </Box>
        );
    }

    // Otherwise show the combined results/time/chat page and include a home/back button
    return (
        <Box sx={{ minHeight: '100vh', bgcolor: '#f5f5f5', py: 3 }}>
            <Container maxWidth="xl" sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                    <Button startIcon={<ArrowBackOutlined />} variant="text" onClick={() => router.push('/?view=home')}>Back</Button>
                    <Typography variant="h5" sx={{ fontWeight: 600 }}>
                        Condition Detail
                    </Typography>
                </Box>

                <Grid container spacing={3}>
                    {/* Left Column: Recommendation (Results) - showActions=false so no bottom buttons */}
                    <Grid item xs={12} md={4}>
                        <ResultsPanel
                            selectedCondition={selectedCondition}
                            showActions={false}
                        />
                    </Grid>

                    {/* Center Column: Time Tracking */}
                    <Grid item xs={12} md={4}>
                        <TimeTrackingPanel conditionId={selectedCondition?.id} />
                    </Grid>

                    {/* Right Column: Chat */}
                    <Grid item xs={12} md={4}>
                        <ChatPanel conditionId={selectedCondition?.id} />
                    </Grid>
                </Grid>
            </Container>
            <AddDiseaseFlow open={showAddFlow} onClose={() => setShowAddFlow(false)} onSaved={handleAddSaved} />
        </Box>
    );
}
