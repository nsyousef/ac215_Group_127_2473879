'use client';

import { createContext, useContext, useEffect, useState } from 'react';
import DiseaseService from '@/services/diseaseService';
import FileAdapter from '@/services/adapters/fileAdapter';

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

  // Add a disease to in-memory state and persist if possible
  const addDisease = async (disease) => {
    // Append to current list
    setDiseases((prev) => {
      const next = [...prev, disease];
      // Try to persist via FileAdapter (Electron) but don't block UI
      (async () => {
        try {
          if (FileAdapter && FileAdapter.saveDiseases) {
            await FileAdapter.saveDiseases(next);
          }
        } catch (e) {
          // ignore persistence errors for now
          console.warn('Failed to persist diseases', e);
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
