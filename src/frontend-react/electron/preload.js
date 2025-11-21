/**
 * Preload script for secure IPC communication between renderer and main process.
 * Exposes a minimal API to the renderer process for file-based operations.
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Open external links in system browser
  openExternal: (url) => ipcRenderer.invoke('open-external', url),

  // Reset all app data (legacy Electron appData directory)
  resetAppData: () => ipcRenderer.invoke('reset-app-data'),

  // ================= ML IPC =================
  mlGetInitialPrediction: (caseId, imagePath, textDescription, userTimestamp) =>
    ipcRenderer.invoke('ml:getInitialPrediction', { caseId, imagePath, textDescription, userTimestamp }),
  mlChatMessage: (caseId, question, userTimestamp) =>
    ipcRenderer.invoke('ml:chatMessage', { caseId, question, userTimestamp }),
  mlSaveBodyLocation: (caseId, bodyLocation) =>
    ipcRenderer.invoke('ml:saveBodyLocation', { caseId, bodyLocation }),
  mlLoadConversationHistory: (caseId) =>
    ipcRenderer.invoke('ml:loadConversationHistory', { caseId }),

  // ================= Demographics IPC =================
  saveDemographics: (demographics) => ipcRenderer.invoke('data:saveDemographics', demographics),
  loadDemographics: () => ipcRenderer.invoke('data:loadDemographics'),
  resetPythonData: () => ipcRenderer.invoke('data:resetPythonData'),

  // ================= Diseases IPC (Python-backed) =================
  loadDiseasesFromPython: () => ipcRenderer.invoke('data:loadDiseases'),
  saveDiseasesToPython: (diseases) => ipcRenderer.invoke('data:saveDiseases', diseases),

  // ================= Case History IPC (Python-backed) =================
  loadCaseHistoryFromPython: (caseId) => ipcRenderer.invoke('data:loadCaseHistory', caseId),
  saveCaseHistoryToPython: (caseId, caseHistory) => ipcRenderer.invoke('data:saveCaseHistory', caseId, caseHistory),
  addTimelineEntry: (caseId, imagePath, note, date) => ipcRenderer.invoke('data:addTimelineEntry', caseId, imagePath, note, date),

  // ================= File Upload IPC =================
  saveUploadedImage: (caseId, filename, buffer) => ipcRenderer.invoke('save-uploaded-image', caseId, filename, buffer),
  readImageAsDataUrl: (imagePath) => ipcRenderer.invoke('read-image-as-data-url', imagePath),
});
