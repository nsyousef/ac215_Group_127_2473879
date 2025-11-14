'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import {
    Box,
    AppBar,
    Toolbar,
    IconButton,
    BottomNavigation,
    BottomNavigationAction,
    Typography,
    Fab,
    useTheme,
} from '@mui/material';
import {
    Home as HomeIcon,
    MapOutlined,
    ArrowBackOutlined,
    AddOutlined,
    PersonOutline,
} from '@mui/icons-material';

export default function MobileLayout({ children, currentPage = 'list', showBottomNav = true, showBackButton = false, onBack, onAddDisease, showProfileIcon = false }) {
    const router = useRouter();
    const searchParams = useSearchParams();
    const viewParam = searchParams.get('view') || 'home';

    // Map view param to tab index: home/list -> 0, map -> 1
    const value = viewParam === 'map' ? 1 : 0;

    const handleNavigation = (newValue) => {
        if (newValue === 0) {
            router.push('/?view=home');
        } else if (newValue === 1) {
            router.push('/?view=map');
        }
    };

    const theme = useTheme();
    const primary = theme.palette.primary.main;
    const bg = theme.palette.background.default;
    const paper = theme.palette.background.paper;

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', bgcolor: bg }}>
            {/* App Bar */}
            <AppBar position="static" sx={{ bgcolor: paper, color: theme.palette.text.primary, boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                <Toolbar>
                    {showBackButton && (
                        <IconButton
                            onClick={onBack}
                            sx={{ mr: 1, color: '#000' }}
                        >
                            <ArrowBackOutlined />
                        </IconButton>
                    )}
                    <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 600 }}>
                        {showBackButton ? 'Details' : 'Home'}
                    </Typography>
                    {showProfileIcon && (
                        <IconButton
                            onClick={() => router.push('/?view=profile')}
                            sx={{ color: '#000' }}
                        >
                            <PersonOutline />
                        </IconButton>
                    )}
                </Toolbar>
            </AppBar>

            {/* Main Content */}
            <Box sx={{ flex: 1, pb: showBottomNav ? 7 : 0, overflow: 'auto', minHeight: 0, WebkitOverflowScrolling: 'touch' }}>
                {children}
            </Box>

            {/* Bottom Navigation - only show on home/map screens */}
            {showBottomNav && (
                <BottomNavigation
                    value={value}
                    onChange={(event, newValue) => handleNavigation(newValue)}
                    sx={{
                        position: 'fixed',
                        bottom: 0,
                        left: 0,
                        right: 0,
                        bgcolor: paper,
                        borderTop: '1px solid #e0e0e0',
                        '& .MuiBottomNavigationAction-root': {
                            color: '#999',
                            '&.Mui-selected': {
                                color: primary,
                            },
                        },
                    }}
                >
                    <BottomNavigationAction label="List" icon={<HomeIcon />} />
                    <BottomNavigationAction label="Body Map" icon={<MapOutlined />} />
                </BottomNavigation>
            )}

            {/* Floating Action Button - only show on home/map screens */}
            {showBottomNav && onAddDisease && (
                <Fab
                    color="primary"
                    onClick={onAddDisease}
                    sx={{
                        position: 'fixed',
                        bottom: 80, // Above bottom nav
                        right: 16,
                        bgcolor: primary,
                        '&:hover': {
                            bgcolor: '#067891',
                        },
                    }}
                >
                    <AddOutlined />
                </Fab>
            )}
        </Box>
    );
}
