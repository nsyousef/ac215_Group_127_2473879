'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
    Box,
    Container,
    Card,
    Grid,
    useMediaQuery,
    useTheme,
    Typography,
    Button,
    IconButton,
} from '@mui/material';
import { ArrowBackOutlined, PersonOutline } from '@mui/icons-material';
import BodyMapView from '@/components/BodyMapView';
import ConditionListView from '@/components/ConditionListView';
import ResultsPanel from '@/components/ResultsPanel';
import TimeTrackingPanel from '@/components/TimeTrackingPanel';
import ChatPanel from '@/components/ChatPanel';
import MobileLayout from '@/components/layouts/MobileLayout';
import AddDiseaseFlow from '@/components/AddDiseaseFlow';
import AddTimeEntryFlow from '@/components/AddTimeEntryFlow';
import ProfilePage from '@/components/ProfilePage';
import OnboardingFlow from '@/components/OnboardingFlow';
import { useDiseaseContext } from '@/contexts/DiseaseContext';
import { useProfile } from '@/contexts/ProfileContext';

export default function PageContent({ initialView }) {
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('md'));
    const router = useRouter();
    const searchParams = useSearchParams();

    // Get view from URL query params (reactive to URL changes)
    const viewParam = searchParams.get('view') || initialView || 'home';
    const conditionIdParam = searchParams.get('conditionId');
    const selectedConditionId = conditionIdParam && conditionIdParam.length > 0 ? `${conditionIdParam}` : null;

    const { diseases, loading: diseasesLoading } = useDiseaseContext();
    const { profile, loading: profileLoading } = useProfile();

    const selectedCondition = useMemo(() => {
        if (!selectedConditionId) return null;
        return (diseases || []).find((d) => `${d.id}` === selectedConditionId) || null;
    }, [diseases, selectedConditionId]);

    const [chatRefreshKey, setChatRefreshKey] = useState(0); // Key to force ChatPanel reload
    const [mobileView, setMobileView] = useState(viewParam);
    const [previousMobileView, setPreviousMobileView] = useState('home'); // Track previous view for back navigation

    const updateQueryParams = useCallback((updates, { replace = false } = {}) => {
        const current = new URLSearchParams(searchParams.toString());
        Object.entries(updates).forEach(([key, value]) => {
            if (value === null || value === undefined || value === '') {
                current.delete(key);
            } else {
                current.set(key, value);
            }
        });
        const queryString = current.toString();
        const nextUrl = queryString ? `/?${queryString}` : '/';
        if (replace) {
            router.replace(nextUrl);
        } else {
            router.push(nextUrl);
        }
    }, [router, searchParams]);

    const setConditionIdInQuery = useCallback((conditionId, { view, replace = true } = {}) => {
        const normalizedId = conditionId === null || conditionId === undefined || conditionId === ''
            ? null
            : `${conditionId}`;
        const updates = { conditionId: normalizedId };
        if (view) {
            updates.view = view;
        }
        updateQueryParams(updates, { replace });
    }, [updateQueryParams]);

    const setView = useCallback((view, { replace = false } = {}) => {
        if (!view) return;
        updateQueryParams({ view }, { replace });
    }, [updateQueryParams]);

    // Sync mobileView with view param and track previous view
    useEffect(() => {
        // Before updating mobileView, save the current view as previous (for detail screens)
        if (['home', 'map'].includes(mobileView)) {
            setPreviousMobileView(mobileView);
        }
        setMobileView(viewParam);
    }, [viewParam]);

    // Clear an invalid conditionId once diseases finish loading
    useEffect(() => {
        if (!diseasesLoading && selectedConditionId && !selectedCondition) {
            setConditionIdInQuery(null, { replace: true });
        }
    }, [diseasesLoading, selectedConditionId, selectedCondition, setConditionIdInQuery]);

    const handleSelectCondition = (condition) => {
        // Handle null condition (e.g., when deleting)
        if (!condition) {
            setConditionIdInQuery(null, { replace: true });
            return;
        }

        // Force ChatPanel to reload conversation history
        setChatRefreshKey((k) => k + 1);

        // Navigate immediately with the updated conditionId persisted in the URL
        const targetView = isMobile ? 'results' : 'home';
        setConditionIdInQuery(condition.id, { view: targetView, replace: true });
    };

    const handleSpotClick = (conditionId) => {
        // Called when a spot is clicked on the body map
        // Works on both mobile (with popover) and desktop
        const exists = (diseases || []).some((c) => c.id === conditionId);
        if (!exists) return;

        // Force ChatPanel to reload conversation history
        setChatRefreshKey((k) => k + 1);
        setConditionIdInQuery(conditionId, { replace: true });
    };

    const handlePopoverViewResults = (condition) => {
        // When user clicks "View Results" in the popover (mobile only)
        if (!condition) return;
        // Force ChatPanel to reload conversation history
        setChatRefreshKey((k) => k + 1);
        setConditionIdInQuery(condition.id, { view: 'results', replace: true });
    };

    const handleBackFromResults = () => {
        // Go back to the previous view
        // If on results, go back to the previous home/map view
        // If on chat/time, go back to results
        // If on profile, go back to home
        if (mobileView === 'results') {
            setView(previousMobileView, { replace: false });
        } else if (['chat', 'time'].includes(mobileView)) {
            setView('results', { replace: false });
        } else if (mobileView === 'profile') {
            setView('home', { replace: false });
        }
    };

    const handleOpenTime = (condition) => {
        const id = condition?.id || selectedConditionId;
        if (!id) return;
        setConditionIdInQuery(id, { view: 'time', replace: false });
    };

    const handleOpenChat = (condition) => {
        const id = condition?.id || selectedConditionId;
        if (!id) return;
        setConditionIdInQuery(id, { view: 'chat', replace: false });
    };

    const handleAddDisease = () => {
        setShowAddFlow(true);
    };

    // Add time entry modal state + refresh key
    const [showAddTimeFlow, setShowAddTimeFlow] = useState(false);
    const [timeEntriesVersion, setTimeEntriesVersion] = useState(0);

    const handleOpenAddTime = (conditionId) => {
        // Accept an optional conditionId from the caller (TimeTrackingPanel passes its conditionId)
        const id = conditionId || selectedConditionId;
        if (!id) return; // require a selected condition

        if (id !== selectedConditionId) {
            setConditionIdInQuery(id, { replace: true });
        }

        setShowAddTimeFlow(true);
    };

    const handleTimeSaved = (entry) => {
        // bump version so TimeTrackingPanel reloads
        setTimeEntriesVersion((v) => v + 1);
        setShowAddTimeFlow(false);
        // Navigate to time view on mobile
        if (isMobile && selectedConditionId) {
            setConditionIdInQuery(selectedConditionId, { view: 'time', replace: false });
        } else if (isMobile) {
            setView('time', { replace: false });
        }
    };

    // Add disease modal state
    const [showAddFlow, setShowAddFlow] = useState(false);

    const handleAddSaved = async (newDisease) => {
        // Close modal
        setShowAddFlow(false);

        // newDisease already contains all enriched fields from Python
        // (description, bodyPart, mapPosition, llmResponse, timelineData, conversationHistory)
        // AND it's already been added to the diseases array by AddDiseaseFlow's addDisease() call
        // So just navigate with the query param as the source of truth!

        // Trigger refresh of TimeTrackingPanel to show initial image
        setTimeEntriesVersion((v) => v + 1);

        // Navigate to results view
        setConditionIdInQuery(newDisease.id, { view: 'results', replace: true });
    };

    const handleStartAnalysis = (tempDisease) => {
        // Called when analysis starts - immediately show results/chat
        console.log('handleStartAnalysis called in PageContent.jsx', tempDisease);

        const targetView = isMobile ? 'chat' : 'results';
        setConditionIdInQuery(tempDisease.id, { view: targetView, replace: true });
    };

    // Render onboarding flow if not completed
    if (!profileLoading && !profile?.hasCompletedOnboarding) {
        return (
            <OnboardingFlow
                onComplete={async (newDisease) => {
                    // newDisease already contains all enriched fields from Python
                    // AND it's already in the diseases array (added by OnboardingFlow's addDisease call)
                    setConditionIdInQuery(newDisease.id, { view: 'results', replace: true });
                }}
            />
        );
    }

    // ========== MOBILE LAYOUT ==========
    if (isMobile) {
        // Show bottom nav only on home/map screens; show back button on detail screens
        const showBottomNav = ['home', 'map'].includes(mobileView);
        const showBackButton = ['results', 'chat', 'time', 'profile'].includes(mobileView);
        const showProfileIcon = ['home', 'map'].includes(mobileView);

        return (
            <MobileLayout
                currentPage={mobileView === 'home' ? 'list' : mobileView === 'map' ? 'body-map' : 'results'}
                showBottomNav={showBottomNav}
                showBackButton={showBackButton}
                showProfileIcon={showProfileIcon}
                onBack={handleBackFromResults}
                onAddDisease={handleAddDisease}
            >
                <Container maxWidth="sm" sx={{ py: 2, pb: showBottomNav ? 9 : 2 }}>
                    {/* Profile View */}
                    {mobileView === 'profile' && (
                        <ProfilePage onBack={() => setView('home', { replace: false })} />
                    )}

                    {/* Home/List View */}
                    {mobileView === 'home' && (
                        <>
                            <Typography variant="body2" sx={{ color: '#666', textAlign: 'center', mb: 2 }}>
                                Select a condition to view details.
                            </Typography>
                            <ConditionListView
                                selectedConditionId={selectedConditionId}
                                onChange={handleSelectCondition}
                                onAddDisease={handleAddDisease}
                                showAddButton={false}
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
                                    selectedConditionId={selectedConditionId}
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
                            <TimeTrackingPanel conditionId={selectedConditionId} onAddImage={handleOpenAddTime} refreshKey={timeEntriesVersion} />
                        </>
                    )}

                    {/* Chat View */}
                    {mobileView === 'chat' && (
                        <>
                            <ChatPanel conditionId={selectedConditionId} refreshKey={chatRefreshKey} />
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
                <AddDiseaseFlow open={showAddFlow} onClose={() => setShowAddFlow(false)} onSaved={handleAddSaved} onStartAnalysis={handleStartAnalysis} />
                <AddTimeEntryFlow open={showAddTimeFlow} onClose={() => setShowAddTimeFlow(false)} conditionId={selectedConditionId} onSaved={handleTimeSaved} />
            </MobileLayout>
        );
    }

    // ========== DESKTOP LAYOUT ==========
    // If the desktop view param is 'home' show the home/list+map layout.
    if (viewParam === 'home') {
        return (
            <Box sx={{ height: '100vh', bgcolor: '#f5f5f5', py: 3, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                <Container maxWidth="xl" sx={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                        <Typography variant="h5" sx={{ fontWeight: 600 }}>
                            Home
                        </Typography>
                        <IconButton onClick={() => setView('profile', { replace: false })} size="large">
                            <PersonOutline />
                        </IconButton>
                    </Box>

                    <Grid container spacing={3} sx={{ flex: 1, minHeight: 0 }}>
                        {/* Left: Conditions list */}
                        <Grid item xs={12} md={4} sx={{ height: '100%', minHeight: 0 }}>
                            <ConditionListView
                                selectedConditionId={selectedConditionId}
                                onChange={handleSelectCondition}
                                onAddDisease={handleAddDisease}
                                showAddButton={true}
                            />
                        </Grid>

                        {/* Center: Body map */}
                        <Grid item xs={12} md={4} sx={{ height: '100%', minHeight: 0 }}>
                            <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
                                <Card sx={{ flex: 1, p: 2, display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 0 }}>
                                    <BodyMapView
                                        selectedConditionId={selectedConditionId}
                                        onSpotClick={handleSpotClick}
                                        maxWidth="420px"
                                        showPopover={false}
                                    />
                                </Card>
                                {/* optional footer space if needed */}
                                <Box sx={{ height: 12 }} />
                            </Box>
                        </Grid>

                        {/* Right: Results */}
                        <Grid item xs={12} md={4} sx={{ height: '100%', minHeight: 0, overflow: 'auto' }}>
                            <ResultsPanel
                                selectedCondition={selectedCondition}
                                showActions={true}
                                onCombined={() => {
                                    if (selectedConditionId) {
                                        setConditionIdInQuery(selectedConditionId, { view: 'results', replace: false });
                                    } else {
                                        setView('results', { replace: false });
                                    }
                                }}
                            />
                        </Grid>
                    </Grid>
                </Container>
                <AddDiseaseFlow open={showAddFlow} onClose={() => setShowAddFlow(false)} onSaved={handleAddSaved} onStartAnalysis={handleStartAnalysis} />
            </Box>
        );
    }

    // Desktop profile view
    if (viewParam === 'profile') {
        return (
            <Box sx={{ minHeight: '100vh', bgcolor: '#f5f5f5', py: 3 }}>
                <Container maxWidth="sm" sx={{ mb: 3 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                        <Button startIcon={<ArrowBackOutlined />} variant="text" onClick={() => setView('home', { replace: false })}>Back</Button>
                        <Typography variant="h5" sx={{ fontWeight: 600 }}>
                            Profile
                        </Typography>
                    </Box>
                    <ProfilePage onBack={() => setView('home', { replace: false })} />
                </Container>
            </Box>
        );
    }

    // Otherwise show the combined results/time/chat page and include a home/back button
    return (
        <Box sx={{ height: '100vh', bgcolor: '#f5f5f5', py: 3, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <Container maxWidth="xl" sx={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                    <Button startIcon={<ArrowBackOutlined />} variant="text" onClick={() => setView('home', { replace: false })}>Back</Button>
                    <Typography variant="h5" sx={{ fontWeight: 600 }}>
                        Condition Detail
                    </Typography>
                </Box>

                <Grid container spacing={3} sx={{ flex: 1, minHeight: 0 }}>
                    {/* Left Column: Recommendation (Results) - showActions=false so no bottom buttons */}
                    <Grid item xs={12} md={4} sx={{ height: '100%', minHeight: 0 }}>
                        <ResultsPanel
                            selectedCondition={selectedCondition}
                            showActions={false}
                        />
                    </Grid>

                    {/* Center Column: Chat */}
                    <Grid item xs={12} md={4} sx={{ height: '100%', minHeight: 0 }}>
                        <ChatPanel conditionId={selectedConditionId} refreshKey={chatRefreshKey} selectedCondition={selectedCondition} />
                    </Grid>

                    {/* Right Column: Time Tracking */}
                    <Grid item xs={12} md={4} sx={{ height: '100%', minHeight: 0 }}>
                        <TimeTrackingPanel conditionId={selectedConditionId} onAddImage={handleOpenAddTime} refreshKey={timeEntriesVersion} />
                    </Grid>
                </Grid>
            </Container>
            <AddDiseaseFlow open={showAddFlow} onClose={() => setShowAddFlow(false)} onSaved={handleAddSaved} />
            <AddTimeEntryFlow open={showAddTimeFlow} onClose={() => setShowAddTimeFlow(false)} conditionId={selectedConditionId} onSaved={handleTimeSaved} />
        </Box>
    );
}
