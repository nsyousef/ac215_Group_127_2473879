'use client';

import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { DiseaseProvider } from '@/contexts/DiseaseContext';
import { ProfileProvider } from '@/contexts/ProfileContext';

const theme = createTheme({
  palette: {
    primary: {
      main: '#0891B2', // brand primary
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#50B4D1', // brand secondary
      contrastText: '#394150',
    },
    background: {
      default: '#E5E7EB', // main app background (swapped as requested)
      paper: '#FFFFFF', // card/toolbars background
    },
    text: {
      primary: '#394150', // dark text default
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
  components: {
    MuiButton: {
      styleOverrides: {
        containedPrimary: {
          backgroundColor: '#0891B2',
          color: '#FFFFFF',
          '&:hover': {
            backgroundColor: '#067891',
          },
        },
        containedSecondary: {
          backgroundColor: '#50B4D1',
          color: '#394150',
          '&:hover': {
            backgroundColor: '#3da6bf',
          },
        },
        textSecondary: {
          color: '#50B4D1',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        colorDefault: {
          backgroundColor: '#FFFFFF',
          color: '#394150',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: '#FFFFFF',
        },
      },
    },
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
