const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Open external links in system browser
  openExternal: (url) => ipcRenderer.invoke('open-external', url),

  // Reset all app data (legacy Electron appData directory)
  resetAppData: () => ipcRenderer.invoke('reset-app-data'),

  // ================= ML IPC =================
  mlGetInitialPrediction: (caseId, imagePath, textDescription, userTimestamp) =>
    ipcRenderer.invoke('ml:getInitialPrediction', {
      caseId,
      imagePath,
      textDescription,
      userTimestamp,
    }),

  // Subscribe to streaming chunks produced during ml:getInitialPrediction
  // Usage in renderer:
  //   const unsubscribe = window.electronAPI.mlOnStreamChunk((chunk) => { ...append to UI... });
  //   // later: unsubscribe();
  mlGetInitialPredictionStream: (caseId, imagePath, textDescription, userTimestamp) => {
    const streamId = `ml-init-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    ipcRenderer.send('ml:getInitialPredictionStream:start', {
      streamId,
      caseId,
      imagePath,
      textDescription,
      userTimestamp,
    });

    const asyncIterator = {
      [Symbol.asyncIterator]() {
        return this;
      },
      next() {
        return new Promise((resolve, reject) => {
          const onChunk = (_event, payload) => {
            if (!payload || payload.streamId !== streamId) return;

            const { done, chunk, error, finalResponse } = payload;

            if (error) {
              cleanup();
              reject(new Error(error));
              return;
            }

            if (done) {
              cleanup();
              // we can surface finalResponse as the last value if you like
              resolve({ value: finalResponse, done: true });
              return;
            }

            resolve({ value: chunk, done: false });
          };

          const cleanup = () => {
            ipcRenderer.removeListener('ml:getInitialPredictionStream:chunk', onChunk);
          };

          ipcRenderer.once('ml:getInitialPredictionStream:chunk', onChunk);
        });
      },
      return() {
        ipcRenderer.send('ml:getInitialPredictionStream:cancel', { streamId });
        return Promise.resolve({ value: undefined, done: true });
      },
    };

    return asyncIterator;
  },
  mlOnStreamChunk: (handler) => {
    if (typeof handler !== 'function') return () => {};

    const listener = (_event, payload) => {
      if (!payload || typeof payload.chunk !== 'string') return;
      handler(payload.chunk);
    };

    ipcRenderer.on('ml:streamChunk', listener);

    // Unsubscribe helper
    return () => {
      ipcRenderer.removeListener('ml:streamChunk', listener);
    };
  },
  mlOnPredictionText: (handler) => {
    if (typeof handler !== 'function') return () => {};

    const listener = (_event, payload) => {
      if (!payload || typeof payload.predictionText !== 'string') return;
      handler(payload.predictionText);
    };

    ipcRenderer.on('ml:predictionText', listener);

    // Unsubscribe helper
    return () => {
      ipcRenderer.removeListener('ml:predictionText', listener);
    };
  },


  mlChatMessage: (caseId, question, userTimestamp) =>
    ipcRenderer.invoke('ml:chatMessage', { caseId, question, userTimestamp }),

  mlSaveBodyLocation: (caseId, bodyLocation) =>
    ipcRenderer.invoke('ml:saveBodyLocation', { caseId, bodyLocation }),

  mlLoadConversationHistory: (caseId) =>
    ipcRenderer.invoke('ml:loadConversationHistory', { caseId }),

  /**
   * Streaming chat API (follow-up questions).
   */
  mlChatMessageStream: (caseId, question, userTimestamp) => {
    const streamId = `ml-stream-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    ipcRenderer.send('ml:chatMessageStream:start', {
      streamId,
      caseId,
      question,
      userTimestamp,
    });

    const asyncIterator = {
      [Symbol.asyncIterator]() {
        return this;
      },
      next() {
        return new Promise((resolve, reject) => {
          const onChunk = (_event, payload) => {
            if (!payload || payload.streamId !== streamId) return;

            const { done, chunk, error } = payload;

            if (error) {
              cleanup();
              reject(new Error(error));
              return;
            }

            if (done) {
              cleanup();
              resolve({ value: undefined, done: true });
              return;
            }

            resolve({ value: chunk, done: false });
          };

          const cleanup = () => {
            ipcRenderer.removeListener('ml:chatMessageStream:chunk', onChunk);
          };

          ipcRenderer.once('ml:chatMessageStream:chunk', onChunk);
        });
      },
      return() {
        ipcRenderer.send('ml:chatMessageStream:cancel', { streamId });
        return Promise.resolve({ value: undefined, done: true });
      },
    };

    return asyncIterator;
  },

  // ================= Demographics IPC =================
  saveDemographics: (demographics) =>
    ipcRenderer.invoke('data:saveDemographics', demographics),
  loadDemographics: () =>
    ipcRenderer.invoke('data:loadDemographics'),
  resetPythonData: () =>
    ipcRenderer.invoke('data:resetPythonData'),

  // ================= Diseases IPC (Python-backed) =================
  loadDiseasesFromPython: () =>
    ipcRenderer.invoke('data:loadDiseases'),
  saveDiseasesToPython: (diseases) =>
    ipcRenderer.invoke('data:saveDiseases', diseases),

  // ================= Case History IPC (Python-backed) =================
  loadCaseHistoryFromPython: (caseId) =>
    ipcRenderer.invoke('data:loadCaseHistory', caseId),
  saveCaseHistoryToPython: (caseId, caseHistory) =>
    ipcRenderer.invoke('data:saveCaseHistory', caseId, caseHistory),
  addTimelineEntry: (caseId, imagePath, note, date) =>
    ipcRenderer.invoke('data:addTimelineEntry', caseId, imagePath, note, date),
  deleteCases: (caseIds) =>
    ipcRenderer.invoke('data:deleteCases', caseIds),

  // ================= File Upload IPC =================
  saveUploadedImage: (caseId, filename, buffer) =>
    ipcRenderer.invoke('save-uploaded-image', caseId, filename, buffer),
  readImageAsDataUrl: (imagePath) =>
    ipcRenderer.invoke('read-image-as-data-url', imagePath),
});
