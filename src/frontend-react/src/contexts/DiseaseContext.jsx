'use client';

import { createContext, useContext, useEffect, useState } from 'react';
import DiseaseService from '@/services/diseaseService';

const DiseaseContext = createContext({
  diseases: [],
  loading: false,
  error: null,
  reload: async () => {},
  addDisease: async () => {},
});

export function useDiseaseContext() {
  return useContext(DiseaseContext);
}

export function DiseaseProvider({ children }) {
  const [diseases, setDiseases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await DiseaseService.loadDiseases();
      setDiseases(data || []);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  // Add a disease to in-memory state and persist via Python
  const addDisease = async (disease) => {
    // Append to current list
    setDiseases((prev) => {
      const next = [...prev, disease];
      // Persist via Python (non-blocking)
      (async () => {
        try {
          if (window.electronAPI?.saveDiseasesToPython) {
            await window.electronAPI.saveDiseasesToPython(next);
          }
        } catch (e) {
          console.warn('Failed to persist diseases to Python', e);
        }
      })();
      return next;
    });
  };

  return (
    <DiseaseContext.Provider value={{ diseases, loading, error, reload: load, addDisease }}>
      {children}
    </DiseaseContext.Provider>
  );
}

export default DiseaseContext;
