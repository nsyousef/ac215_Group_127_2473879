/**
 * Preload script for secure IPC communication between renderer and main process.
 * Exposes a minimal API to the renderer process for file-based operations.
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Load all diseases from file system
  loadDiseases: () => ipcRenderer.invoke('load-diseases'),

  // Load chat history for a specific condition
  loadChatHistory: (conditionId) => ipcRenderer.invoke('load-chat-history', conditionId),

  // Load time tracking entries for a specific condition
  loadTimeTracking: (conditionId) => ipcRenderer.invoke('load-time-tracking', conditionId),

  // Save a new chat message (future use)
  saveChatMessage: (conditionId, message) => ipcRenderer.invoke('save-chat-message', conditionId, message),

  // Save a new time tracking entry (future use)
  saveTimeEntry: (conditionId, entry) => ipcRenderer.invoke('save-time-entry', conditionId, entry),

  // Save diseases list
  saveDiseases: (diseases) => ipcRenderer.invoke('save-diseases', diseases),

  // Load profile data
  loadProfile: () => ipcRenderer.invoke('load-profile'),

  // Save profile data
  saveProfile: (profile) => ipcRenderer.invoke('save-profile', profile),

  // Get the app data directory path
  getAppDataPath: () => ipcRenderer.invoke('get-app-data-path'),

  // Open external links in system browser
  openExternal: (url) => ipcRenderer.invoke('open-external', url),

  // Reset all app data
  resetAppData: () => ipcRenderer.invoke('reset-app-data'),
});
