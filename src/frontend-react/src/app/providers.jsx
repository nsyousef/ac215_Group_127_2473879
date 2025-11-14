'use client';

import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { DiseaseProvider } from '@/contexts/DiseaseContext';
import { ProfileProvider } from '@/contexts/ProfileContext';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});

export function Providers({ children }) {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ProfileProvider>
        <DiseaseProvider>
          {children}
        </DiseaseProvider>
      </ProfileProvider>
    </ThemeProvider>
  );
}
