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
      const hasCoin = metadata.hasCoin !== undefined ? metadata.hasCoin : false;
      console.log('[mlClient] getInitialPrediction - metadata:', metadata, 'hasCoin:', hasCoin);
      
      try {
        const result = await window.electronAPI.mlGetInitialPrediction(
          caseId,
          imagePath,
          textDescription,
          userTimestamp,
          hasCoin,
        );
        
        // Check if this is an UNCERTAIN response (resolved, not rejected, to avoid console error)
        // Return it as-is so the calling code can check for isUncertain flag
        if (result && result.isUncertain) {
          return result;
        }
        
        return result;
      } catch (error) {
        // Clean up Electron IPC error prefix before rethrowing (for other errors)
        const cleanMessage = error.message.replace(/^Error invoking remote method '[^']+': Error: /, '');
        const cleanError = new Error(cleanMessage);
        cleanError.code = error.code;
        cleanError.details = error.details;
        cleanError.stack = error.stack;
        throw cleanError;
      }
    }

    // Non-Electron fallback: no public dummy loading; return a structured error
    throw new Error('ML prediction is only available in Electron runtime.');
  }

  /**
   * Send follow-up chat message (non-streaming).
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
  async *getInitialPredictionStream(imagePath, textDescription, caseId, metadata = {}) {
    await this.initialize();
    const userTimestamp = new Date().toISOString();
    const hasCoin = metadata.hasCoin !== undefined ? metadata.hasCoin : false;

    if (isElectron() && window.electronAPI?.mlGetInitialPredictionStream) {
      const stream = window.electronAPI.mlGetInitialPredictionStream(
        caseId,
        imagePath,
        textDescription,
        userTimestamp,
        hasCoin,
      );

      for await (const item of stream) {
        // During streaming, item is usually a string chunk
        // On final step (done=true), we passed finalResponse as value
        yield item;
      }
      return;
    }

    // Fallback: non-streaming; just return final result once
    const result = await this.getInitialPrediction(imagePath, textDescription, caseId, metadata);
    yield { finalResponse: result };
  }

  async chatMessage(caseId, userQuery) {
    await this.initialize();

    if (isElectron() && window.electronAPI && window.electronAPI.mlChatMessage) {
      const userTimestamp = new Date().toISOString();
      return await window.electronAPI.mlChatMessage(caseId, userQuery, userTimestamp);
    }
    throw new Error('Chat is only available in Electron runtime.');
  }

  /**
   * Stream follow-up chat response in chunks.
   *
   * Expected Electron IPC shape:
   *   window.electronAPI.mlChatMessageStream(caseId, userQuery, userTimestamp)
   *     -> returns an async iterable of chunks.
   *
   * Each yielded `chunk` can be:
   *   - a string, or
   *   - an object like { delta: '...', done: false } / { text: '...' }
   *
   * @param {string} caseId - Unique case identifier
   * @param {string} userQuery - User's follow-up question
   * @returns {AsyncGenerator<string|Object>} Async iterable of chunks
   */
  async *chatMessageStream(caseId, userQuery) {
    await this.initialize();
    const userTimestamp = new Date().toISOString();

    // Electron + streaming IPC path
    if (isElectron() && window.electronAPI && window.electronAPI.mlChatMessageStream) {
      // Assumes mlChatMessageStream returns an async iterable
      const stream = window.electronAPI.mlChatMessageStream(
        caseId,
        userQuery,
        userTimestamp,
      );

      // If preload returns an async iterable, forward chunks directly
      if (stream && typeof stream[Symbol.asyncIterator] === 'function') {
        for await (const chunk of stream) {
          yield chunk;
        }
        return;
      }

      // If the API returns a single response object instead of a stream,
      // normalize to one final chunk.
      if (stream && stream.answer) {
        yield stream.answer;
        return;
      }
    }

    // Fallback: use non-streaming chatMessage and yield once
    const fullResponse = await this.chatMessage(caseId, userQuery);
    if (fullResponse && fullResponse.answer) {
      yield fullResponse.answer;
    } else {
      // In case the shape is different or answer is missing,
      // yield the whole object so the caller can inspect it.
      yield fullResponse;
    }
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
