// Lightweight disease service with adapter selection
import { CONDITIONS } from '@/lib/constants';
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
        if (data && data.length) return data;
      } catch (e) {
        // fall back to bundled constants
        console.warn('FileAdapter failed, falling back to bundled CONDITIONS', e);
      }
    }

    // Default: return bundled constants (synchronous, but wrap in Promise)
    return Promise.resolve(CONDITIONS);
  },
};

export default DiseaseService;
