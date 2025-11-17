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
    
    def __init__(self, case_id: str, dummy: bool = False):
        """
        Initialize APIManager for a specific case.
        
        Args:
            case_id: Unique identifier for the case
            dummy: If True, use dummy data for all operations (for frontend testing)
        """
        self.case_id = case_id
        self.dummy = dummy
        
        # Set up storage directories
        self.history_dir = Path("history")
        self.conversation_dir = Path("conversations")
        self.history_dir.mkdir(exist_ok=True)
        self.conversation_dir.mkdir(exist_ok=True)
        
        # File paths
        self.history_file = self.history_dir / f"{case_id}.json"
        self.conversation_file = self.conversation_dir / f"{case_id}.json"
        
        # Load existing data
        self.history = self._load_history(case_id)
        
        self.llm_explain_url = "https://tanushkmr2001--dermatology-llm-27b-dermatologyllm-explain-dev.modal.run"
        self.llm_followup_url = "https://tanushkmr2001--dermatology-llm-27b-dermatologyllm-as-a085de-dev.modal.run"
    
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
        
        # Step 1: Run local ML model for embeddings TODO
        print("  → Running local ML model for embeddings...")
        embedding = self._run_local_ml_model(image)
        
        # Step 2: Run CV analysis TODO
        print("  → Running CV analysis...")
        cv_analysis = self._run_cv_analysis(image)
        
        # Step 3: Get predictions from cloud ML model TODO
        print("  → Getting cloud predictions...")
        predictions = self._run_cloud_ml_model(embedding, text_description)
        
        # Step 4: Build metadata for LLM
        metadata = {
            "user_input": text_description,
            "cv_analysis": cv_analysis,
            "history": self.history
        }
        
        # Step 5: Call LLM API for explanation
        print("  → Calling LLM for explanation...")
        llm_response = self._call_llm_explain(predictions, metadata)
        
        # Step 6: Save to conversation history (initial LLM response)
        print("  → Saving conversation...")
        self._save_conversation_entry(
            case_id=case_id,
            user_message=text_description,
            llm_response=llm_response,
            is_initial=True
        )
        
        # Step 7: Save to history (CV analysis, predictions, image path)
        print("  → Saving history...")
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        self._save_history_entry(
            case_id=case_id,
            date=current_date,
            cv_analysis=cv_analysis,
            predictions=predictions,
            image_path=str(image) if image else None
        )
        
        # Step 8: Prepare complete results
        results = {
            "llm_response": llm_response,
            "predictions": predictions,
            "cv_analysis": cv_analysis,
            "embedding": embedding,
            "text_description": text_description
        }
        
        print(f"✓ Initial prediction complete for case {case_id}")
        return results
    
    def chat_message(self, case_id: str, user_query: str) -> Dict[str, Any]:
        """
        Handle follow-up chat messages from the user.
        
        Workflow:
        1. Load conversation history
        2. Get initial answer from first conversation entry
        3. Call LLM API with user's question
        4. Save updated conversation
        5. Return response
        
        Args:
            case_id: Unique identifier for the case
            user_query: User's follow-up question
            
        Returns:
            Dictionary containing:
                - answer: LLM's response to the question
                - conversation_history: All conversation entries
        """
        if self.dummy:
            print(f"[DUMMY MODE] Using dummy chat message for case {case_id}")
            return {
                "answer": "This is a dummy response for frontend testing",
                "conversation_history": []
            }
        
        print(f"Processing chat message for case {case_id}...")
        
        # Step 1: Load conversation history
        print("  → Loading conversation history...")
        conversation_data = self._load_conversation_history(case_id)
        
        if not conversation_data or len(conversation_data) == 0:
            raise ValueError(
                f"No conversation found for case {case_id}. "
                "Please run get_initial_prediction first."
            )
        
        # Step 2: Get initial answer from first entry
        initial_answer = conversation_data[0].get("llm_response", "")
        
        # Build conversation history for context (list of previous questions)
        conversation_history = [entry["user_message"] for entry in conversation_data[1:]]
        
        # Step 3: Call LLM API for follow-up
        print("  → Calling LLM for follow-up answer...")
        response = self._call_llm_followup(
            initial_answer=initial_answer,
            question=user_query,
            conversation_history=conversation_history
        )
        
        # Step 4: Save new conversation entry
        print("  → Saving conversation...")
        self._save_conversation_entry(
            case_id=case_id,
            user_message=user_query,
            llm_response=response["answer"],
            is_initial=False
        )
        
        print(f"✓ Chat message processed for case {case_id}")
        return {
            "answer": response["answer"],
            "conversation_history": response["conversation_history"]
        }
    
    # ==================== Helper Methods ====================
    
    def _load_history(self, case_id: str) -> Dict[str, Any]:
        """
        Load case history from disk.
        
        History structure:
        {
            "dates": {
                "2025-11-15": {
                    "cv_analysis": {...},
                    "predictions": {...},
                    "image_path": "path/to/image.jpg"
                }
            }
        }
        
        Args:
            case_id: Unique identifier for the case
            
        Returns:
            Dictionary containing case history
        """
        if self.dummy:
            print(f"[DUMMY MODE] Using dummy history for case {case_id}")
            return {
                "dates": {
                    "2025-10-10": {
                        "cv_analysis": {
                            "area": 12.6,
                            "color_profile": {
                                "average_Lab": [63.5, 22.1, 10.8],
                                "redness_index": 0.46,
                                "texture_contrast": 0.17
                            }
                        },
                        "predictions": {
                            "eczema": 0.82,
                            "contact_dermatitis": 0.12
                        },
                        "image_path": "images/case_001_2025-10-10.jpg"
                    },
                    "2025-10-20": {
                        "cv_analysis": {
                            "area": 9.8,
                            "color_profile": {
                                "average_Lab": [65.8, 20.0, 10.1],
                                "redness_index": 0.38,
                                "texture_contrast": 0.15
                            }
                        },
                        "predictions": {
                            "eczema": 0.75,
                            "contact_dermatitis": 0.18
                        },
                        "image_path": "images/case_001_2025-10-20.jpg"
                    }
                }
            }
        
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse history file for case {case_id}")
                return {"dates": {}}
        else:
            print(f"No history found for case {case_id}, starting fresh")
            return {"dates": {}}
    
    def _save_results(self, case_id: str, data: Dict[str, Any]) -> None:
        """
        Save case data to disk.
        
        Args:
            case_id: Unique identifier for the case
            data: Dictionary of data to save
        """
        if self.dummy:
            print(f"  [DUMMY MODE] Skipping save for case {case_id}")
            return
        
        history_file = self.history_dir / f"{case_id}.json"
        
        # TODO: Implement browser storage for frontend integration
        # TODO: Add error handling for file write failures
        
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  Saved data to {history_file}")
    
    def _run_local_ml_model(self, image: Any) -> List[float]:
        """
        Run local ML model for embeddings.
        
        TODO: Integrate with actual ML model from src/ml_workflow/
        TODO: Handle different image input formats (PIL, numpy, file path)
        TODO: Add proper error handling
        
        Args:
            image: Image data (format TBD)
            
        Returns:
            Image embedding vector (list of floats)
        """
        if self.dummy:
            print("    [DUMMY MODE] Returning dummy embeddings...")
            return [0.1, 0.2, 0.3, 0.4] * 128  # Dummy 512-dim embedding
        
        # TODO: REAL IMPLEMENTATION
        print("    [TODO] Running local ML model for embeddings...")
        return [0.1, 0.2, 0.3, 0.4] * 128  # Placeholder
    
    def _run_cv_analysis(self, image: Any) -> Dict[str, Any]:
        """
        Run computer vision analysis on the image.
        
        TODO: Integrate with actual CV analysis from src/ml_workflow/
        TODO: Handle different image input formats (PIL, numpy, file path)
        TODO: Add proper error handling
        
        Args:
            image: Image data (format TBD)
            
        Returns:
            Dictionary containing CV analysis:
                - area: Lesion area
                - color_profile: Color metrics (Lab values, redness, etc.)
                - boundary_irregularity: Shape metrics
                - symmetry_score: Symmetry metrics
        """
        if self.dummy:
            print("    [DUMMY MODE] Returning dummy CV analysis...")
            return {
                "area": 8.4,
                "color_profile": {
                    "average_Lab": [67.2, 18.4, 9.3],
                    "redness_index": 0.34,
                    "texture_contrast": 0.12
                },
                "boundary_irregularity": 0.23,
                "symmetry_score": 0.78
            }
        
        # TODO: REAL IMPLEMENTATION
        print("    [TODO] Running CV analysis...")
        return {
            "area": 8.4,
            "color_profile": {
                "average_Lab": [67.2, 18.4, 9.3],
                "redness_index": 0.34,
                "texture_contrast": 0.12
            },
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
        if self.dummy:
            print("    [DUMMY MODE] Returning dummy predictions...")
            return {
                "eczema": 0.78,
                "contact_dermatitis": 0.15,
                "psoriasis": 0.04,
                "tinea_corporis": 0.02,
                "seborrheic_dermatitis": 0.01
            }
        
        # TODO: REAL IMPLEMENTATION
        print("    [TODO] Calling cloud ML model...")
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
        if self.dummy:
            print("    [DUMMY MODE] Returning dummy LLM explanation...")
            return (
                "It sounds like you might be dealing with eczema, which is a common "
                "skin condition that causes red, itchy patches. Based on what you've "
                "described and the analysis, the affected area shows signs of inflammation "
                "with increased redness and texture changes typical of eczematous skin.\n\n"
                "[DUMMY RESPONSE - This is test data for frontend integration]"
            )
        
        # TODO: REAL IMPLEMENTATION
        print("    [TODO] Calling LLM explain API...")
        try:
            response = requests.post(
                self.llm_explain_url,
                json={
                    "predictions": predictions,
                    "metadata": metadata
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()  # or response.text depending on API format
        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM API: {e}")
            raise
    
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
        if self.dummy:
            print("    [DUMMY MODE] Returning dummy LLM followup...")
            updated_history = conversation_history + [question]
            return {
                "answer": (
                    f"That's a great question about: '{question}'. "
                    "[DUMMY RESPONSE - This is test data for frontend integration]"
                ),
                "conversation_history": updated_history
            }
        
        # TODO: REAL IMPLEMENTATION
        print("    [TODO] Calling LLM followup API...")
        try:
            response = requests.post(
                self.llm_followup_url,
                json={
                    "initial_answer": initial_answer,
                    "question": question,
                    "conversation_history": conversation_history
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM API: {e}")
            raise
    
    def _save_conversation_entry(
        self,
        case_id: str,
        user_message: str,
        llm_response: str,
        is_initial: bool = False
    ) -> None:
        """
        Save a conversation entry to disk.
        
        Args:
            case_id: Unique identifier for the case
            user_message: User's message/question
            llm_response: LLM's response
            is_initial: Whether this is the initial conversation entry
        """
        if self.dummy:
            print(f"    [DUMMY MODE] Skipping save conversation for case {case_id}")
            return
        
        conversation_file = self.conversation_dir / f"{case_id}.json"
        
        # Load existing conversation
        conversation_data = []
        if conversation_file.exists():
            try:
                with open(conversation_file, 'r') as f:
                    conversation_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse conversation file for case {case_id}")
        
        # Append new entry
        entry = {
            "user_message": user_message,
            "llm_response": llm_response,
            "timestamp": Path(conversation_file).stat().st_mtime if conversation_file.exists() else None,
            "is_initial": is_initial
        }
        conversation_data.append(entry)
        
        # Save
        with open(conversation_file, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        print(f"    Saved conversation entry to {conversation_file}")
    
    def _save_history_entry(
        self,
        case_id: str,
        date: str,
        cv_analysis: Dict[str, Any],
        predictions: Dict[str, float],
        image_path: Optional[str] = None
    ) -> None:
        """
        Save a history entry to disk.
        
        Args:
            case_id: Unique identifier for the case
            date: Date of the entry (YYYY-MM-DD)
            cv_analysis: CV analysis results
            predictions: Disease predictions
            image_path: Optional path to image
        """
        if self.dummy:
            print(f"    [DUMMY MODE] Skipping save history for case {case_id}")
            return
        
        history_file = self.history_dir / f"{case_id}.json"
        
        # Load existing history
        history_data = {"dates": {}}
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse history file for case {case_id}")
        
        # Add new entry
        history_data["dates"][date] = {
            "cv_analysis": cv_analysis,
            "predictions": predictions,
            "image_path": image_path
        }
        
        # Save
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"    Saved history entry to {history_file}")
    
    def _load_conversation_history(self, case_id: str) -> List[Dict[str, Any]]:
        """
        Load conversation history from disk.
        
        Args:
            case_id: Unique identifier for the case
            
        Returns:
            List of conversation entries
        """
        if self.dummy:
            print(f"    [DUMMY MODE] Returning empty conversation for case {case_id}")
            return []
        
        conversation_file = self.conversation_dir / f"{case_id}.json"
        
        if conversation_file.exists():
            try:
                with open(conversation_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse conversation file for case {case_id}")
                return []
        else:
            print(f"No conversation found for case {case_id}")
            return []
