/**
 * FileAdapter: Communicates with Electron main process via IPC to load/save data.
 * This adapter bridges the renderer process to the file system through secure IPC.
 */

const FileAdapter = {
  // Load diseases from file system via IPC
  async load() {
    if (!window.electronAPI) {
      console.warn('Electron API not available; cannot load diseases from file.');
      return null;
    }

    try {
      const diseases = await window.electronAPI.loadDiseases();
      return diseases; // null if file not found, array if found
    } catch (e) {
      console.error('FileAdapter.load() error:', e);
      throw e;
    }
  },

  // Load chat history for a specific condition
  async loadChat(conditionId) {
    if (!window.electronAPI) {
      console.warn('Electron API not available; cannot load chat history.');
      return [];
    }

    try {
      return await window.electronAPI.loadChatHistory(conditionId);
    } catch (e) {
      console.error(`FileAdapter.loadChat(${conditionId}) error:`, e);
      return []; // Return empty on error
    }
  },

  // Load time tracking entries for a specific condition
  async loadTimeTracking(conditionId) {
    if (!window.electronAPI) {
      console.warn('Electron API not available; cannot load time tracking.');
      return [];
    }

    try {
      return await window.electronAPI.loadTimeTracking(conditionId);
    } catch (e) {
      console.error(`FileAdapter.loadTimeTracking(${conditionId}) error:`, e);
      return []; // Return empty on error
    }
  },

  // Save a chat message (future use)
  async saveChat(conditionId, message) {
    if (!window.electronAPI) {
      console.warn('Electron API not available; cannot save chat message.');
      return null;
    }

    try {
      return await window.electronAPI.saveChatMessage(conditionId, message);
    } catch (e) {
      console.error(`FileAdapter.saveChat(${conditionId}) error:`, e);
      throw e;
    }
  },

  // Save a time tracking entry (future use)
  async saveTimeEntry(conditionId, entry) {
    if (!window.electronAPI) {
      console.warn('Electron API not available; cannot save time entry.');
      return null;
    }

    try {
      return await window.electronAPI.saveTimeEntry(conditionId, entry);
    } catch (e) {
      console.error(`FileAdapter.saveTimeEntry(${conditionId}) error:`, e);
      throw e;
    }
  },

  // Save diseases array to disk (Electron)
  async saveDiseases(diseases) {
    if (!window.electronAPI) {
      console.warn('Electron API not available; cannot save diseases.');
      return null;
    }

    try {
      return await window.electronAPI.saveDiseases(diseases);
    } catch (e) {
      console.error('FileAdapter.saveDiseases() error:', e);
      throw e;
    }
  },
};

export default FileAdapter;
