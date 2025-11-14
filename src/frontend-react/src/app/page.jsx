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
    const { profile, loading: profileLoading, updateProfile } = useProfile();

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
        // If on profile, go back to home
        if (mobileView === 'results') {
            router.push(`/?view=${previousMobileView}`);
        } else if (['chat', 'time'].includes(mobileView)) {
            router.push('/?view=results');
        } else if (mobileView === 'profile') {
            router.push('/?view=home');
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

    // Add time entry modal state + refresh key
    const [showAddTimeFlow, setShowAddTimeFlow] = useState(false);
    const [timeEntriesVersion, setTimeEntriesVersion] = useState(0);

    const handleOpenAddTime = (conditionId) => {
        // Accept an optional conditionId from the caller (TimeTrackingPanel passes its conditionId)
        const id = conditionId || selectedCondition?.id;
        if (!id) return; // require a selected condition

        // If the caller provided an id that's different from the current selection, update it
        if (!selectedCondition || selectedCondition.id !== id) {
            const found = (diseases || []).find((d) => d.id === id);
            if (found) setSelectedCondition(found);
        }

        setShowAddTimeFlow(true);
    };

    const handleTimeSaved = (entry) => {
        // bump version so TimeTrackingPanel reloads
        setTimeEntriesVersion((v) => v + 1);
        setShowAddTimeFlow(false);
        // Navigate to time view on mobile
        if (isMobile) router.push('/?view=time');
    };

    // Add disease modal state
    const [showAddFlow, setShowAddFlow] = useState(false);

    const handleAddSaved = (newDisease) => {
        // Close modal and select condition, navigate to results/detail
        setShowAddFlow(false);
        setSelectedCondition(newDisease);
        router.push('/?view=results');
    };

    // Render onboarding flow if not completed
    if (!profileLoading && !profile?.hasCompletedOnboarding) {
        return (
            <OnboardingFlow
                onComplete={(newDisease) => {
                    // Set selected condition and navigate appropriately
                    setSelectedCondition(newDisease);
                    if (isMobile) {
                        router.push('/?view=results');
                    } else {
                        router.push('/?view=results');
                    }
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
                        <ProfilePage onBack={() => router.push('/?view=home')} />
                    )}

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
                            <TimeTrackingPanel conditionId={selectedCondition?.id} onAddImage={handleOpenAddTime} refreshKey={timeEntriesVersion} />
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
                <AddTimeEntryFlow open={showAddTimeFlow} onClose={() => setShowAddTimeFlow(false)} conditionId={selectedCondition?.id} onSaved={handleTimeSaved} />
            </MobileLayout>
        );
    }

    // ========== DESKTOP LAYOUT ==========
    // If the desktop view param is 'home' show the home/list+map layout.
    if (viewParam === 'home') {
        return (
            <Box sx={{ minHeight: '100vh', bgcolor: '#f5f5f5', py: 3 }}>
                <Container maxWidth="xl" sx={{ mb: 3 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                        <Typography variant="h5" sx={{ fontWeight: 600 }}>
                            Home
                        </Typography>
                        <IconButton onClick={() => router.push('/?view=profile')} size="large">
                            <PersonOutline />
                        </IconButton>
                    </Box>

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

    // Desktop profile view
    if (viewParam === 'profile') {
        return (
            <Box sx={{ minHeight: '100vh', bgcolor: '#f5f5f5', py: 3 }}>
                <Container maxWidth="sm" sx={{ mb: 3 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                        <Button startIcon={<ArrowBackOutlined />} variant="text" onClick={() => router.push('/?view=home')}>Back</Button>
                        <Typography variant="h5" sx={{ fontWeight: 600 }}>
                            Profile
                        </Typography>
                    </Box>
                    <ProfilePage onBack={() => router.push('/?view=home')} />
                </Container>
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
                        <TimeTrackingPanel conditionId={selectedCondition?.id} onAddImage={handleOpenAddTime} refreshKey={timeEntriesVersion} />
                    </Grid>

                    {/* Right Column: Chat */}
                    <Grid item xs={12} md={4}>
                        <ChatPanel conditionId={selectedCondition?.id} />
                    </Grid>
                </Grid>
            </Container>
            <AddDiseaseFlow open={showAddFlow} onClose={() => setShowAddFlow(false)} onSaved={handleAddSaved} />
            <AddTimeEntryFlow open={showAddTimeFlow} onClose={() => setShowAddTimeFlow(false)} conditionId={selectedCondition?.id} onSaved={handleTimeSaved} />
        </Box>
    );
}
