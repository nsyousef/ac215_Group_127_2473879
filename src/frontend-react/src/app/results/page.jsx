'use client';

import {
    Box,
    Container,
    Card,
    CardContent,
    CardMedia,
    Button,
    Typography,
    ButtonGroup,
} from '@mui/material';
import {
    AssignmentReturnOutlined,
    QuestionAnswerOutlined,
} from '@mui/icons-material';
import Image from 'next/image';

export default function Results() {
    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', bgcolor: '#f5f5f5' }}>
            {/* Header */}
            <Box sx={{ bgcolor: '#fff', py: 2, boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                <Container maxWidth="sm">
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        Results
                    </Typography>
                </Container>
            </Box>

            {/* Main Content */}
            <Box sx={{ flex: 1, pb: 2, overflow: 'auto' }}>
                <Container maxWidth="sm" sx={{ py: 2 }}>
                    {/* Image Card */}
                    <Card sx={{ mb: 2 }}>
                        <Box sx={{ position: 'relative', width: '100%', paddingBottom: '100%' }}>
                            <Image
                                src="/public/assets/nasty_skin.jpg"
                                alt="Skin condition"
                                fill
                                style={{ objectFit: 'cover' }}
                            />
                        </Box>
                    </Card>

                    {/* Recommendation Card */}
                    <Card sx={{ mb: 2 }}>
                        <CardContent>
                            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                                Recommendation:
                            </Typography>
                            <Typography variant="body2" sx={{ color: '#666', mb: 2 }}>
                                Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
                                incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud
                                exercitation ullamco laboris nisl ut aliquip ex ea commodo consequat.
                            </Typography>
                            <Typography variant="body2" sx={{ color: '#666' }}>
                                Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat
                                nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia
                                deserunt mollit anim id est laborum.
                            </Typography>
                        </CardContent>
                    </Card>

                    {/* Action Buttons */}
                    <ButtonGroup
                        variant="contained"
                        fullWidth
                        sx={{
                            '& > button': {
                                flex: 1,
                                py: 1.5,
                                textTransform: 'none',
                                fontSize: '0.9rem',
                            },
                        }}
                    >
                        <Button
                            startIcon={<AssignmentReturnOutlined />}
                            sx={{ bgcolor: '#1976d2' }}
                        >
                            Track Progress
                        </Button>
                        <Button
                            startIcon={<QuestionAnswerOutlined />}
                            sx={{ bgcolor: '#1976d2' }}
                        >
                            Ask Question
                        </Button>
                    </ButtonGroup>
                </Container>
            </Box>
        </Box>
    );
}
