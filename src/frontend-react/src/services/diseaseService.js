// Lightweight disease service with adapter selection
import { isElectron } from '@/utils/config';
// File adapter is a stub for future electron/file-system based loading
import FileAdapter from './adapters/fileAdapter';

const DiseaseService = {
  // loadDiseases returns a Promise resolving to an array of disease objects
  async loadDiseases() {
    // In Electron we may load from file system via FileAdapter
    if (isElectron()) {
      try {
        const data = await FileAdapter.load();
        if (Array.isArray(data)) return data; // May be empty
      } catch (e) {
        console.warn('FileAdapter failed to load diseases', e);
      }
    }
    // Do NOT load bundled dummy diseases anymore; return empty list
    return Promise.resolve([]);
  },
};

export default DiseaseService;
