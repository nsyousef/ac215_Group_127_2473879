// Lightweight disease service - loads from Python in Electron
import { isElectron } from '@/utils/config';

const DiseaseService = {
  // loadDiseases returns a Promise resolving to an array of disease objects
  async loadDiseases() {
    // In Electron, load from Python
    if (isElectron()) {
      try {
        if (window.electronAPI?.loadDiseasesFromPython) {
          const data = await window.electronAPI.loadDiseasesFromPython();
          // Handle both array response and potential {diseases: [...]} object
          const diseases = Array.isArray(data) ? data : (data?.diseases || []);
          if (diseases.length > 0) return diseases;
        }
      } catch (e) {
        console.warn('Failed to load diseases from Python', e);
      }
    }
    // Return empty list if not in Electron or loading failed
    return Promise.resolve([]);
  },
};

export default DiseaseService;
