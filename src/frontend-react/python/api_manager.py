import json
import os
import shutil
from copy import deepcopy
import datetime
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests

SAVE_DIR = Path(os.getcwd())

def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

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
        self.case_dir = SAVE_DIR / case_id
        self.case_dir.mkdir(exist_ok=True)
        print(self.case_dir)
        
        # File paths
        self.case_history_file = self.case_dir / f"case_history.json"
        self.conversation_file = self.case_dir / f"conversation_history.json"
        self.demographics_file = SAVE_DIR / "demographics.json"
        
        # Load existing data
        self.case_history = self._load_json_file(
            file_path=self.case_history_file,
            # Case history structure: dict with keys 'dates' and 'location'
            # dates structure: dict with keys 'cv_analysis' and 'image_path' and 'text_summary' and 'predictions
            # location structure: {"location": {"coordinates": (x,y), "nlp": "elbow"}}
            default_value={"dates": {}, "location": {}},
            description="case history"
        )
        self.conversation_history = self._load_json_file(
            file_path=self.conversation_file,
            # conversation history structure: List of dicts
            # Each dict has {'user': {'message': str, 'timestamp': str}, 'llm': {'message': str, 'timestamp': str}}
            default_value=[],
            description="conversation history"
        )
        self.demographics = self._load_json_file(
            file_path=self.demographics_file,
            default_value={},
            description="demographics"
        )
        
        self.llm_explain_url = "https://tanushkmr2001--dermatology-llm-27b-dermatologyllm-explain.modal.run"
        self.llm_followup_url = "https://tanushkmr2001--dermatology-llm-27b-dermatologyllm-ask-followup.modal.run"
    
    def get_initial_prediction(
        self, 
        image: Any, 
        text_description: str, 
        user_timestamp: Optional[str] = None,
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
        print(f"Processing initial prediction for case {self.case_id}...")
        # Step 0: Save image to case directory #COMPLETED
        print("  → Saving image...") 
        image_path = self._save_image(image)
        
        # Step 1: Run local ML model for embeddings TODO
        print("  → Running local ML model for embeddings...")
        embedding = self._run_local_ml_model(image_path)
        
        # Step 2: Run CV analysis TODO
        print("  → Running CV analysis...")
        cv_analysis = self._run_cv_analysis(image_path)
        
        # Step 3: Get predictions from cloud ML model TODO
        print("  → Getting cloud predictions...")
        updated_text_description = self.update_text_input(text_description)
        predictions = self._run_cloud_ml_model(embedding, updated_text_description)
        
        # Step 4: Build metadata for LLM
        metadata = {
            "user_input": updated_text_description,
            "cv_analysis": cv_analysis,
            "history": self.case_history['dates']
        }
        
        # Step 5: Call LLM API for explanation
        print("  → Calling LLM for explanation...")
        llm_response, llm_timestamp = self._call_llm_explain(predictions, metadata)
        
        # Step 6: Save to conversation history (initial LLM response)
        print("  → Saving conversation...")
        self._save_conversation_entry(
            user_message=updated_text_description,
            llm_response=llm_response,
            user_timestamp=user_timestamp,
            llm_timestamp=llm_timestamp
        )
        
        # Step 7: Save to history (CV analysis, predictions, image path)
        print("  → Saving history...")
        current_date = datetime.now().strftime("%Y-%m-%d")
        self._save_history_entry(
            date=current_date,
            cv_analysis=cv_analysis,
            predictions=predictions,
            image_path=image_path,
            text_summary=updated_text_description,
        )
        
        # Step 8: Prepare complete results
        results = {
            "llm_response": llm_response,
            "predictions": predictions,
            "cv_analysis": cv_analysis,
            "embedding": embedding,
            "text_description": text_description
        }
        
        print(f"✓ Initial prediction complete for case {self.case_id}")
        return results

    def update_text_input(self, text_input: str) -> str:
        """
        Augment user-provided text input with demographic details.
        """
        if not isinstance(text_input, str):
            text_input = str(text_input)

        demographic_sentences = []
        demographics = self.demographics or {}

        # DOB → age sentence if possible
        dob = demographics.get("DOB")
        if dob:
            try:
                from datetime import datetime, date

                birth_date = datetime.strptime(dob, "%Y-%m-%d").date()
                today = date.today()
                age = today.year - birth_date.year - (
                    (today.month, today.day) < (birth_date.month, birth_date.day)
                )
                demographic_sentences.append(f"The age is {age}.")
            except ValueError:
                demographic_sentences.append(f"The date of birth is {dob}.")

        sex = demographics.get("Sex")
        if sex:
            demographic_sentences.append(f"Sex is {sex}.")

        race = demographics.get("Race")
        if race:
            demographic_sentences.append(f"Race is {race}.")

        country = demographics.get("Country")
        if country:
            demographic_sentences.append(f"Country is {country}.")
        location_nlp = (
            (self.case_history or {}).get("location", {}).get("nlp")
            if isinstance(self.case_history, dict)
            else None
        )
        if location_nlp:
            demographic_sentences.append(
                f"The body location of the affected area is {location_nlp}."
            )

        if demographic_sentences:
            text_input = text_input.strip()
            text_input = f"{text_input} {' '.join(demographic_sentences)}"

        return text_input
    
    def chat_message(self, user_query: str, user_timestamp: Optional[str] = None) -> Dict[str, Any]:
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

        print(f"Processing chat message for case {self.case_id}...")
        
        # Step 1: Load conversation history
        print("  → Loading conversation history...")
        conversation_data = self._load_json_file(
            file_path=self.conversation_file,
            default_value=[],
            description="conversation history"
        )
        
        if not conversation_data or len(conversation_data) == 0:
            raise ValueError(
                f"No conversation found for case {self.case_id}. "
                "Please run get_initial_prediction first."
            )
        
        # Step 2: Extract initial message and build conversation history
        initial_message = conversation_data[0]
        conversation_history = conversation_data[1:]  # All entries after first
        
        # Step 3: Call LLM API for follow-up
        print("  → Calling LLM for follow-up answer...")
        response, llm_timestamp = self._call_llm_followup(
            initial_message=initial_message,
            question=user_query,
            conversation_history=conversation_history[-5:]  # Last 5 entries
        )
        
        # Step 4: Save new conversation entry
        print("  → Saving conversation...")
        self._save_conversation_entry(
            user_message=user_query,
            llm_response=response["answer"],
            user_timestamp=user_timestamp,
            llm_timestamp=llm_timestamp
        )
        
        print(f"✓ Chat message processed for case {self.case_id}")
        return {
            "answer": response["answer"],
            "conversation_history": response["conversation_history"]
        }
    
    # ==================== Helper Methods ====================
    def _save_conversation_entry(
        self,
        user_message: str,
        llm_response: str,
        user_timestamp: Optional[str],
        llm_timestamp: Optional[str],
    ) -> None:
        """
        Persist conversation turns using the structured format:
        {
            "user": {"message": str, "timestamp": str},
            "llm": {"message": str, "timestamp": str}
        }
        """
        entry = {
            "user": {
                "message": user_message,
                "timestamp": str(user_timestamp) if user_timestamp else _timestamp(),
            },
            "llm": {
                "message": llm_response,
                "timestamp": str(llm_timestamp) if llm_timestamp else _timestamp(),
            },
        }

        if not isinstance(self.conversation_history, list):
            self.conversation_history = []

        self.conversation_history.append(entry)

        try:
            with open(self.conversation_file, "w") as f:
                json.dump(self.conversation_history, f, indent=2)
        except OSError as exc:
            print(f"Error saving conversation history: {exc}")

    def _save_history_entry(
        self,
        date: str,
        cv_analysis: Dict[str, Any],
        predictions: Dict[str, Any],
        image_path: Optional[str],
        text_summary: Optional[str] = None,
    ) -> None:
        """
        Persist a history entry following the case history schema.
        """

        dates = self.case_history.setdefault("dates", {})
        dates[date] = {
            "cv_analysis": cv_analysis,
            "predictions": predictions,
            "image_path": image_path,
            "text_summary": text_summary or "",
        }

        try:
            with open(self.case_history_file, "w") as f:
                json.dump(self.case_history, f, indent=2)
        except OSError as exc:
            print(f"Error saving history: {exc}")

    def _save_image(self, image: Any) -> Optional[str]:
        """Save image to the case images folder (sequentially numbered)."""

        images_dir = self.case_dir / "images"
        images_dir.mkdir(exist_ok=True)

        next_idx = len(list(images_dir.glob("image_*.png"))) + 1
        filename = f"image_{next_idx:04d}.png"
        destination = images_dir / filename

        try:
            image.save(destination)
            return str(destination)

        except Exception as exc:
            raise ValueError(f"Error saving image: {exc}")
    

    def _load_json_file(
        self,
        file_path: Path,
        default_value: Any,
        description: str,
    ) -> Any:
        """
        Generic JSON loader with graceful fallbacks.
        """
        if not file_path.exists():
            print(f"No {description} file found for case {self.case_id}, using default.")
            return deepcopy(default_value)

        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {description} file for case {self.case_id}. Using default.")
            return deepcopy(default_value)
    
    def _run_local_ml_model(self, image_path: str) -> List[float]:
        """
        Run local ML model for embeddings.
        
        TODO: Integrate with actual ML model from src/ml_workflow/
        TODO: Handle different image input formats (PIL, numpy, file path)
        TODO: Add proper error handling
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image embedding vector (list of floats)
        """
        if self.dummy:
            print("    [DUMMY MODE] Returning dummy embeddings...")
            return [0.1, 0.2, 0.3, 0.4] * 128  # Dummy 512-dim embedding
        
        # TODO: REAL IMPLEMENTATION
        print("    [TODO] Running local ML model for embeddings...")
        return [0.1, 0.2, 0.3, 0.4] * 128  # Placeholder
    
    def _run_cv_analysis(self, image_path: str) -> Dict[str, Any]:
        """
        Run computer vision analysis on the image.
        
        TODO: Integrate with actual CV analysis from src/ml_workflow/
        TODO: Handle different image input formats (PIL, numpy, file path)
        TODO: Add proper error handling
        
        Args:
            image_path: Path to the image file
            
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
    ) -> Tuple[Any, str]:
        """
        Call LLM API to generate initial explanation.
        
        TODO: Handle API errors and timeouts
        TODO: Add retry logic
        TODO: Validate response format
        
        Args:
            predictions: Disease predictions with confidence scores
            metadata: Additional context (user input, CV analysis, history)
            
        Returns:
            Tuple of (LLM-generated explanation payload, timestamp string)
        """
        
        print(" Calling LLM explain API...")
        try:
            response = requests.post(
                self.llm_explain_url,
                json={
                    "predictions": predictions,
                    "metadata": metadata
                },
                timeout=300
            )
            response.raise_for_status()
            llm_timestamp = _timestamp()
            return response.json(), llm_timestamp  # or response.text depending on API format
        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM API: {e}")
            raise
    
    def _call_llm_followup(
        self,
        initial_message: Dict[str, Any],
        question: str,
        conversation_history: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], str]:
        """
        Call LLM API for follow-up question.
        
        Args:
            initial_message: The first conversation entry (dict with 'user' and 'llm' keys)
            question: User's follow-up question
            conversation_history: List of conversation entry dicts (last 5)
            
        Returns:
            Tuple containing:
                - Dictionary with answer and updated conversation history
                - Timestamp string for the LLM response
        """
        # Extract initial answer from initial_message dict
        initial_answer = initial_message.get("llm", {}).get("message", "")
        
        # Extract user messages from conversation history dicts
        history_questions = [
            entry.get("user", {}).get("message", "")
            for entry in conversation_history
            if entry.get("user", {}).get("message")
        ]
        
        print(" Calling LLM followup API...")
        try:
            response = requests.post(
                self.llm_followup_url,
                json={
                    "initial_answer": initial_answer,
                    "question": question,
                    "conversation_history": history_questions
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json(), _timestamp()
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
