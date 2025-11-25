/**
 * ML Client - Interface to Python API Manager
 *
 * This service handles communication with the Python backend (api_manager.py)
 * for ML predictions and LLM chat functionality.
 *
 * In production, this will use IPC to communicate with the Python process.
 * For now, we simulate the Python API locally with dummy data matching
 * the api_manager.py dummy mode responses.
 */

import { isElectron } from '@/utils/config';

class MLClient {
  constructor() {
    // TODO: Initialize IPC connection to Python process when Electron integration is ready
    this.initialized = false;
  }

  /**
   * Initialize connection to Python backend
   * TODO: Set up IPC handlers for Electron
   */
  async initialize() {
    if (this.initialized) return;
    // TODO: Wait for Python process to be ready
    this.initialized = true;
  }

  /**
   * Save body location before processing prediction
   *
   * @param {string} caseId - Unique case identifier
   * @param {Object} bodyLocation - Object with coordinates and nlp fields
   * @returns {Promise<Object>} Success result
   */
  async saveBodyLocation(caseId, bodyLocation) {
    await this.initialize();

    if (isElectron() && window.electronAPI && window.electronAPI.mlSaveBodyLocation) {
      return await window.electronAPI.mlSaveBodyLocation(caseId, bodyLocation);
    }

    throw new Error('Body location save is only available in Electron runtime.');
  }

  /**
   * Get initial prediction from ML model
   *
   * Workflow (matching api_manager.py):
   * 1. Local vision model generates embeddings from image
   * 2. CV analysis extracts features (area, color, etc.)
   * 3. Cloud ML model returns disease predictions
   * 4. LLM generates explanation from predictions + context
   * 5. Results saved to local storage
   *
   * @param {string} imagePath - File path to uploaded image
   * @param {string} textDescription - User's description of symptoms
   * @param {string} caseId - Unique case identifier
   * @param {Object} metadata - Optional user metadata (dateOfBirth, sex, ethnicity)
   * @returns {Promise<Object>} Results object with predictions, LLM response, etc.
   */
  async getInitialPrediction(imagePath, textDescription, caseId, metadata = {}) {
    await this.initialize();

    if (isElectron() && window.electronAPI && window.electronAPI.mlGetInitialPrediction) {
      const userTimestamp = new Date().toISOString();
      // Route to Python via Electron IPC
      return await window.electronAPI.mlGetInitialPrediction(caseId, imagePath, textDescription, userTimestamp);
    }

    // Non-Electron fallback: no public dummy loading; return a structured error
    throw new Error('ML prediction is only available in Electron runtime.');
  }

  /**
   * Send follow-up chat message
   *
   * Workflow (matching api_manager.py):
   * 1. Load conversation history from local storage
   * 2. Send question + context to LLM
   * 3. Get response from LLM
   * 4. Save updated conversation
   *
   * @param {string} caseId - Unique case identifier
   * @param {string} userQuery - User's follow-up question
   * @returns {Promise<Object>} Response object with answer and conversation history
   */
  async chatMessage(caseId, userQuery) {
    await this.initialize();

    if (isElectron() && window.electronAPI && window.electronAPI.mlChatMessage) {
      const userTimestamp = new Date().toISOString();
      return await window.electronAPI.mlChatMessage(caseId, userQuery, userTimestamp);
    }
    throw new Error('Chat is only available in Electron runtime.');
  }

  /**
   * Load conversation history for a case
   *
   * @param {string} caseId - Unique case identifier
   * @returns {Promise<Array>} Array of conversation entries
   */
  async loadConversation(caseId) {
    await this.initialize();

    // TODO: Replace with actual IPC call to Python
    // For now, return empty array (conversation managed by ChatPanel)
    return [];
  }

  /**
   * Load case history (time tracking entries)
   *
   * @param {string} caseId - Unique case identifier
   * @returns {Promise<Object>} History object with dates and entries
   */
  async loadHistory(caseId) {
    await this.initialize();

    // TODO: Replace with actual IPC call to Python
    // For now, return empty history (managed by FileAdapter)
    return {
      dates: {},
    };
  }
}

// Export singleton instance
const mlClient = new MLClient();
export default mlClient;
