import React from 'react';
import { MahalanobisPlot } from './components/MahalanobisPlot';
import { Container, CssBaseline, ThemeProvider, createTheme } from '@mui/material';

const theme = createTheme();

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <MahalanobisPlot />
      </Container>
    </ThemeProvider>
  );
};

export default App;
