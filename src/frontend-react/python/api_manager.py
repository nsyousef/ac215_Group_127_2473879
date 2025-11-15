"""
API Manager for Dermatology Assistant

This class acts as a messenger between the frontend and various backend services:
- Local ML models (embeddings and CV analysis)
- Cloud prediction APIs
- LLM APIs (Modal deployment)

Author: AC215 Group 127
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests


class APIManager:
    """
    Orchestrates communication between frontend, ML models, and LLM APIs.
    
    This class manages the flow of data from user inputs through various
    processing stages (local ML, cloud predictions, LLM explanations) and
    back to the frontend.
    """
    
    def __init__(self, case_id: str):
        """
        Initialize APIManager for a specific case.
        
        Args:
            case_id: Unique identifier for the case
        """
        self.case_id = case_id
        
        # Storage path for case data
        self.history_dir = Path("history")
        self.history_dir.mkdir(exist_ok=True)
        
        # Load case history
        self.history = self._load_history(case_id)
        
        # TODO: Configure these URLs from environment variables or config file
        self.llm_explain_url = "https://your-modal-url.modal.run/explain"
        self.llm_followup_url = "https://your-modal-url.modal.run/ask_followup"
    
    def get_initial_prediction(
        self, 
        image: Any, 
        text_description: str, 
        case_id: str
    ) -> Dict[str, Any]:
        """
        Process initial image and text input to generate predictions and LLM analysis.
        
        Workflow:
        1. Run local ML model for embeddings and CV analysis
        2. Send embedding to cloud for disease predictions
        3. Build metadata combining all inputs and history
        4. Call LLM API to generate explanation
        5. Save results for future reference
        
        Args:
            image: Image data (format TBD - could be PIL Image, numpy array, or file path)
            text_description: User's description of their skin condition
            case_id: Unique identifier for this case
            
        Returns:
            Dictionary containing:
                - llm_response: Text explanation from LLM
                - predictions: Disease predictions with confidence scores
                - cv_analysis: Computer vision analysis results
                - embedding: Image embedding vector
        """
        print(f"Processing initial prediction for case {case_id}...")
        
        # Step 1: Run local ML model for CV analysis and embeddings
        print("  → Running local ML model...")
        ml_results = self._run_local_ml_model(image)
        embedding = ml_results["embedding"]
        cv_analysis = ml_results["cv_analysis"]
        
        # Step 2: Get predictions from cloud ML model
        print("  → Getting cloud predictions...")
        predictions = self._run_cloud_ml_model(embedding, text_description)
        
        # Step 3: Build metadata for LLM
        metadata = {
            "user_input": text_description,
            "cv_analysis": cv_analysis,
            "history": self.history
        }
        
        # Step 4: Call LLM API for explanation
        print("  → Calling LLM for explanation...")
        llm_response = self._call_llm_explain(predictions, metadata)
        
        # Step 5: Prepare complete results
        results = {
            "llm_response": llm_response,
            "predictions": predictions,
            "cv_analysis": cv_analysis,
            "embedding": embedding,
            "text_description": text_description,
            "initial_answer": llm_response,  # Store for future chat context
            "conversation_history": []  # Initialize empty conversation history
        }
        
        # Step 6: Save results to disk/browser
        print("  → Saving results...")
        self._save_results(case_id, results)
        
        print(f"✓ Initial prediction complete for case {case_id}")
        return results
    
    def chat_message(self, case_id: str, user_query: str) -> Dict[str, Any]:
        """
        Handle follow-up chat messages from the user.
        
        Workflow:
        1. Load case history and previous conversation
        2. Call LLM API with user's question
        3. Update conversation history
        4. Save updated data
        5. Return response
        
        Args:
            case_id: Unique identifier for the case
            user_query: User's follow-up question
            
        Returns:
            Dictionary containing:
                - answer: LLM's response to the question
                - conversation_history: Updated list of questions
        """
        print(f"Processing chat message for case {case_id}...")
        
        # Step 1: Load case history
        print("  → Loading case history...")
        case_data = self._load_history(case_id)
        
        # Extract initial answer and conversation history
        initial_answer = case_data.get("initial_answer", "")
        conversation_history = case_data.get("conversation_history", [])
        
        if not initial_answer:
            raise ValueError(
                f"No initial analysis found for case {case_id}. "
                "Please run get_initial_prediction first."
            )
        
        # Step 2: Call LLM API for follow-up
        print("  → Calling LLM for follow-up answer...")
        response = self._call_llm_followup(
            initial_answer=initial_answer,
            question=user_query,
            conversation_history=conversation_history
        )
        
        # Step 3: Update conversation history
        updated_history = response["conversation_history"]
        
        # Step 4: Save updated conversation
        print("  → Saving conversation...")
        case_data["conversation_history"] = updated_history
        case_data["last_question"] = user_query
        case_data["last_answer"] = response["answer"]
        self._save_results(case_id, case_data)
        
        print(f"✓ Chat message processed for case {case_id}")
        return response
    
    # ==================== Helper Methods ====================
    
    def _load_history(self, case_id: str) -> Dict[str, Any]:
        """
        Load case history from disk.
        
        Args:
            case_id: Unique identifier for the case
            
        Returns:
            Dictionary containing case history, or empty dict if not found
        """
        history_file = self.history_dir / f"{case_id}.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse history file for case {case_id}")
                return {}
        else:
            print(f"No history found for case {case_id}, starting fresh")
            return {}
    
    def _save_results(self, case_id: str, data: Dict[str, Any]) -> None:
        """
        Save case data to disk.
        
        Args:
            case_id: Unique identifier for the case
            data: Dictionary of data to save
        """
        history_file = self.history_dir / f"{case_id}.json"
        
        # TODO: Implement browser storage for frontend integration
        # TODO: Add error handling for file write failures
        
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  Saved data to {history_file}")
    
    def _run_local_ml_model(self, image: Any) -> Dict[str, Any]:
        """
        Run local ML model for embeddings and CV analysis.
        
        TODO: Integrate with actual ML model from src/ml_workflow/
        TODO: Handle different image input formats (PIL, numpy, file path)
        TODO: Add proper error handling
        
        Args:
            image: Image data (format TBD)
            
        Returns:
            Dictionary containing:
                - embedding: Image embedding vector
                - cv_analysis: Computer vision analysis (area, color, texture)
        """
        # DUMMY IMPLEMENTATION
        print("    [DUMMY] Running local ML model...")
        
        return {
            "embedding": [0.1, 0.2, 0.3, 0.4] * 128,  # Dummy 512-dim embedding
            "cv_analysis": {
                "area": 8.4,
                "color_profile": {
                    "average_Lab": [67.2, 18.4, 9.3],
                    "redness_index": 0.34,
                    "texture_contrast": 0.12
                },
                "boundary_irregularity": 0.23,
                "symmetry_score": 0.78
            }
        }
    
    def _run_cloud_ml_model(
        self, 
        embedding: List[float], 
        text_description: str
    ) -> Dict[str, float]:
        """
        Send embedding and text to cloud for disease predictions.
        
        TODO: Integrate with actual cloud ML API
        TODO: Add authentication/API key handling
        TODO: Add retry logic and error handling
        TODO: Handle timeouts gracefully
        
        Args:
            embedding: Image embedding vector from local model
            text_description: User's text description
            
        Returns:
            Dictionary mapping disease names to confidence scores
        """
        # DUMMY IMPLEMENTATION
        print("    [DUMMY] Calling cloud ML model...")
        
        return {
            "eczema": 0.78,
            "contact_dermatitis": 0.15,
            "psoriasis": 0.04,
            "tinea_corporis": 0.02,
            "seborrheic_dermatitis": 0.01
        }
    
    def _call_llm_explain(
        self, 
        predictions: Dict[str, float], 
        metadata: Dict[str, Any]
    ) -> str:
        """
        Call LLM API to generate initial explanation.
        
        TODO: Handle API errors and timeouts
        TODO: Add retry logic
        TODO: Validate response format
        
        Args:
            predictions: Disease predictions with confidence scores
            metadata: Additional context (user input, CV analysis, history)
            
        Returns:
            LLM-generated explanation text
        """
        # DUMMY IMPLEMENTATION
        print("    [DUMMY] Calling LLM explain API...")
        
        # TODO: Uncomment and configure when API is ready
        # try:
        #     response = requests.post(
        #         self.llm_explain_url,
        #         json={
        #             "predictions": predictions,
        #             "metadata": metadata
        #         },
        #         timeout=30
        #     )
        #     response.raise_for_status()
        #     return response.json()  # or response.text depending on API format
        # except requests.exceptions.RequestException as e:
        #     print(f"Error calling LLM API: {e}")
        #     raise
        
        # Dummy response for now
        return (
            "It sounds like you might be dealing with eczema, which is a common "
            "skin condition that causes red, itchy patches. Based on what you've "
            "described and the analysis, the affected area shows signs of inflammation "
            "with increased redness and texture changes typical of eczematous skin.\n\n"
            "[DUMMY RESPONSE - Replace with actual LLM call]"
        )
    
    def _call_llm_followup(
        self,
        initial_answer: str,
        question: str,
        conversation_history: List[str]
    ) -> Dict[str, Any]:
        """
        Call LLM API for follow-up question.
        
        TODO: Handle API errors and timeouts
        TODO: Add retry logic
        TODO: Validate response format
        
        Args:
            initial_answer: The original LLM explanation
            question: User's follow-up question
            conversation_history: List of previous questions in conversation
            
        Returns:
            Dictionary containing:
                - answer: LLM's response
                - conversation_history: Updated conversation history
        """
        # DUMMY IMPLEMENTATION
        print("    [DUMMY] Calling LLM followup API...")
        
        # TODO: Uncomment and configure when API is ready
        # try:
        #     response = requests.post(
        #         self.llm_followup_url,
        #         json={
        #             "initial_answer": initial_answer,
        #             "question": question,
        #             "conversation_history": conversation_history
        #         },
        #         timeout=30
        #     )
        #     response.raise_for_status()
        #     return response.json()
        # except requests.exceptions.RequestException as e:
        #     print(f"Error calling LLM API: {e}")
        #     raise
        
        # Dummy response for now
        updated_history = conversation_history + [question]
        return {
            "answer": (
                f"That's a great question about: '{question}'. "
                "[DUMMY RESPONSE - Replace with actual LLM call]"
            ),
            "conversation_history": updated_history
        }


