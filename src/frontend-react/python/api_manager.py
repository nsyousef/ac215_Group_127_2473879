import json
import os
import sys
import shutil
from copy import deepcopy
import datetime
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests
from inference_local.vision_encoder import VisionEncoder

# Shared vision encoder instance (lazy-loaded)
_VISION_ENCODER = None

# Get app data directory from Electron, fallback to current working directory
APP_DATA_DIR = os.getenv("APP_DATA_DIR")
if APP_DATA_DIR:
    SAVE_DIR = Path(APP_DATA_DIR)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
else:
    SAVE_DIR = Path(os.getcwd())

# Cloud ML API URLs (can be overridden with environment variables)
BASE_URL = os.getenv("BASE_URL", "https://inference-cloud-469023639150.us-east4.run.app")
TEXT_EMBEDDING_URL = os.getenv("TEXT_EMBEDDING_URL", f"{BASE_URL}/embed-text")
PREDICTION_URL = os.getenv("PREDICTION_URL", f"{BASE_URL}/predict")

# LLM API URLs (can be overridden with environment variables for testing)
DEFAULT_LLM_EXPLAIN_URL = "https://tanushkmr2001--dermatology-llm-27b-dermatologyllm-explain.modal.run"
DEFAULT_LLM_FOLLOWUP_URL = "https://tanushkmr2001--dermatology-llm-27b-dermatologyllm-ask-followup.modal.run"


def debug_log(msg: str):
    """Print to stderr so it doesn't interfere with stdout JSON protocol"""
    print(msg, file=sys.stderr, flush=True)


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
        debug_log(str(self.case_dir))

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
            description="case history",
        )
        self.conversation_history = self._load_json_file(
            file_path=self.conversation_file,
            # conversation history structure: List of dicts
            # Each dict has {'user': {'message': str, 'timestamp': str}, 'llm': {'message': str, 'timestamp': str}}
            default_value=[],
            description="conversation history",
        )
        self.demographics = self._load_json_file(
            file_path=self.demographics_file, default_value={}, description="demographics"
        )

        # Set API URLs (can be overridden with environment variables)
        self.llm_explain_url = os.getenv("LLM_EXPLAIN_URL", DEFAULT_LLM_EXPLAIN_URL)
        self.llm_followup_url = os.getenv("LLM_FOLLOWUP_URL", DEFAULT_LLM_FOLLOWUP_URL)
        self.text_embed_url = TEXT_EMBEDDING_URL
        self.prediction_url = PREDICTION_URL

    @staticmethod
    def get_vision_encoder():
        """Get shared vision encoder instance (lazy-loaded)."""
        global _VISION_ENCODER

        if _VISION_ENCODER is None:
            debug_log("Initializing vision encoder...")

            # Look for checkpoint (REQUIRED)
            current_file = Path(__file__).resolve()
            python_dir = current_file.parent
            MODEL_NAME = "test_best.pth"
            checkpoint_path = python_dir / "inference_local" / MODEL_NAME

            if not checkpoint_path.exists():
                error_msg = (
                    f"Model checkpoint not found at {checkpoint_path}\n"
                    f"Please ensure {MODEL_NAME} is placed in python/inference_local/model/"
                )
                debug_log(f"✗ {error_msg}")
                raise FileNotFoundError(error_msg)

            debug_log(f"Using checkpoint: {checkpoint_path}")

            try:
                _VISION_ENCODER = VisionEncoder(checkpoint_path=str(checkpoint_path))
                info = _VISION_ENCODER.get_model_info()
                debug_log(f"✓ Vision encoder initialized: {info}")
            except Exception as e:
                debug_log(f"✗ Error initializing vision encoder: {e}")
                import traceback

                debug_log(traceback.format_exc())
                raise

        return _VISION_ENCODER

    @staticmethod
    def save_demographics(data: Dict[str, Any]) -> None:
        """
        Save demographics data to demographics.json at root level.

        Args:
            data: Dictionary containing demographic information (DOB, Sex, Race, etc.)
        """
        demographics_file = SAVE_DIR / "demographics.json"
        with open(demographics_file, "w") as f:
            json.dump(data, f, indent=2)
        debug_log(f"Saved demographics to {demographics_file}")

    @staticmethod
    def load_demographics() -> Dict[str, Any]:
        """
        Load demographics data from demographics.json.

        Returns:
            Dictionary containing demographic information, or empty dict if not found
        """
        demographics_file = SAVE_DIR / "demographics.json"
        if demographics_file.exists():
            try:
                with open(demographics_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                debug_log(f"Warning: Could not parse demographics file")
                return {}
        return {}

    @staticmethod
    def save_body_location(case_id: str, body_location: Dict[str, Any]) -> None:
        """
        Save body location to case_history.json before APIManager initialization.
        Creates case directory and history file if they don't exist.

        Args:
            case_id: Unique identifier for the case
            body_location: Dict with 'coordinates' [x, y] and 'nlp' (natural language description)
        """
        case_dir = SAVE_DIR / case_id
        case_dir.mkdir(exist_ok=True)

        case_history_file = case_dir / "case_history.json"

        # Load existing history or create new
        if case_history_file.exists():
            try:
                with open(case_history_file, "r") as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                history = {"dates": {}, "location": {}}
        else:
            history = {"dates": {}, "location": {}}

        # Update location
        history["location"] = body_location

        # Save back
        with open(case_history_file, "w") as f:
            json.dump(history, f, indent=2)

        debug_log(f"Saved body location for case {case_id}: {body_location}")

    @staticmethod
    def load_diseases() -> List[Dict[str, Any]]:
        """
        Load diseases list from diseases.json at root level.
        Enriches minimal disease data with information from case_history.json.

        Returns:
            List of disease dictionaries with derived fields, or empty list if not found
        """
        diseases_file = SAVE_DIR / "diseases.json"
        if not diseases_file.exists():
            return []

        try:
            with open(diseases_file, "r") as f:
                diseases = json.load(f)
        except json.JSONDecodeError:
            debug_log(f"Warning: Could not parse diseases file")
            return []

        # Enrich each disease with data from case_history.json
        enriched_diseases = []
        for disease in diseases:
            case_id = disease.get("id")
            if not case_id:
                continue

            # Ensure case_id has proper format (with case_ prefix for folder lookup)
            # If ID already has case_ prefix, use it; otherwise add it
            folder_id = case_id if case_id.startswith("case_") else f"case_{case_id}"

            # Load case history to get additional fields
            case_history = APIManager.load_case_history(folder_id)

            # Get name from case_history, or fall back to disease.name
            name = case_history.get("name", disease.get("name", "Unknown Condition"))

            # Get earliest date entry for description and confidence
            dates = case_history.get("dates", {})
            earliest_date = min(dates.keys()) if dates else None

            description = ""
            confidenceLevel = 0
            if earliest_date:
                entry = dates[earliest_date]
                # Get description from text_summary
                description = entry.get("text_summary", "")[:100] + "..." if entry.get("text_summary", "") else ""
                # Get confidence from top prediction
                predictions = entry.get("predictions", {})
                if predictions:
                    confidenceLevel = int(max(predictions.values()) * 100)

            # Get body part from location
            location = case_history.get("location", {})
            bodyPart = location.get("nlp", "Unknown")
            coordinates = location.get("coordinates", None)
            mapPosition = None
            if coordinates and len(coordinates) == 2:
                mapPosition = {"leftPct": coordinates[0], "topPct": coordinates[1]}

            # Get LLM response from conversation history (first LLM message)
            llmResponse = ""
            conversation_file = SAVE_DIR / folder_id / "conversation_history.json"
            if conversation_file.exists():
                try:
                    with open(conversation_file, "r") as f:
                        conversation = json.load(f)
                        # Get first LLM response
                        if conversation and len(conversation) > 0:
                            first_entry = conversation[0]
                            llmResponse = first_entry.get("llm", {}).get("message", "")
                except Exception as e:
                    debug_log(f"Warning: Could not load conversation for case {case_id}: {e}")

            # Build enriched disease object
            # Always use ID without case_ prefix for frontend consistency
            clean_id = case_id.replace("case_", "") if case_id.startswith("case_") else case_id
            enriched_disease = {
                "id": clean_id,
                "name": name,
                "description": description,
                "bodyPart": bodyPart,
                "mapPosition": mapPosition,
                "image": disease.get("image"),  # Keep thumbnail from diseases.json
                "confidenceLevel": confidenceLevel,
                "date": earliest_date or "",
                "llmResponse": llmResponse,  # Initial LLM explanation
                # Add timeline data from dates
                "timelineData": (
                    [
                        {
                            "id": i,
                            "date": date,
                            "image": dates[date].get("image_path", ""),
                            "note": dates[date].get("text_summary", ""),
                        }
                        for i, date in enumerate(sorted(dates.keys(), reverse=True))
                    ]
                    if dates
                    else []
                ),
            }

            enriched_diseases.append(enriched_disease)

        return enriched_diseases

    @staticmethod
    def save_diseases(diseases: List[Dict[str, Any]]) -> None:
        """
        Save diseases list to diseases.json at root level.
        Only saves minimal data (id, name, image). Other fields are derived from case_history.json.

        Args:
            diseases: List of disease dictionaries (may contain extra fields that will be filtered out)
        """
        diseases_file = SAVE_DIR / "diseases.json"

        # Only save minimal fields to avoid duplication
        minimal_diseases = [
            {
                "id": d.get("id"),
                "name": d.get("name", "Unknown Condition"),
                "image": d.get("image"),  # Thumbnail for quick loading
            }
            for d in diseases
        ]

        with open(diseases_file, "w") as f:
            json.dump(minimal_diseases, f, indent=2)
        debug_log(f"Saved {len(minimal_diseases)} diseases to {diseases_file}")

    @staticmethod
    def load_case_history(case_id: str) -> Dict[str, Any]:
        """
        Load complete case history including dates, location, and name.

        Args:
            case_id: Unique identifier for the case

        Returns:
            Dictionary with 'dates', 'location', and optionally 'name' keys, or default structure if not found
        """
        case_dir = SAVE_DIR / case_id
        case_history_file = case_dir / "case_history.json"

        if case_history_file.exists():
            try:
                with open(case_history_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                debug_log(f"Warning: Could not parse case history file for case {case_id}")
                return {"dates": {}, "location": {}, "name": "Unknown Condition"}
        return {"dates": {}, "location": {}, "name": "Unknown Condition"}

    @staticmethod
    def save_case_history(case_id: str, case_history: Dict[str, Any]) -> None:
        """
        Save complete case history to case_history.json.

        Args:
            case_id: Unique identifier for the case
            case_history: Dictionary containing 'dates', 'location', and optionally 'name' data
        """
        case_dir = SAVE_DIR / case_id
        case_dir.mkdir(exist_ok=True)

        case_history_file = case_dir / "case_history.json"

        with open(case_history_file, "w") as f:
            json.dump(case_history, f, indent=2)

        debug_log(f"Saved case history for case {case_id}")

    @staticmethod
    def update_disease_name(case_id: str, name: str) -> None:
        """
        Update the disease name in case_history.json.

        Args:
            case_id: Case ID (with or without 'case_' prefix)
            name: New disease name
        """
        # Ensure case_id has 'case_' prefix
        if not case_id.startswith("case_"):
            case_id = f"case_{case_id}"

        case_history = APIManager.load_case_history(case_id)
        case_history["name"] = name
        APIManager.save_case_history(case_id, case_history)
        debug_log(f"Updated disease name for case {case_id}: {name}")

    @staticmethod
    def add_timeline_entry(case_id: str, image_path: str, note: str, date: str) -> None:
        """
        Add a new timeline entry (manually uploaded image) to case_history.json.

        Args:
            case_id: Case ID (with or without 'case_' prefix)
            image_path: Path to the saved image file
            note: User's notes/description for this entry
            date: ISO date string (YYYY-MM-DD) for when this entry was added
        """
        # Ensure case_id has 'case_' prefix for folder lookup
        if not case_id.startswith("case_"):
            case_id = f"case_{case_id}"

        # Load existing case history
        case_history = APIManager.load_case_history(case_id)

        # Add new entry to dates using the provided date as key
        if "dates" not in case_history:
            case_history["dates"] = {}

        case_history["dates"][date] = {
            "image_path": image_path,
            "text_summary": note or "",
            "cv_analysis": {},  # Manual entries don't have CV analysis
            "predictions": {},  # Manual entries don't have predictions
        }

        # Save updated case history
        APIManager.save_case_history(case_id, case_history)
        debug_log(f"Added timeline entry for case {case_id} on date {date}")

    @staticmethod
    def reset_all_data() -> None:
        """
        Clear all Python-managed data including demographics, diseases list, and all case directories.
        This is used for the "Reset App" functionality.
        """
        # Remove demographics file
        demographics_file = SAVE_DIR / "demographics.json"
        if demographics_file.exists():
            demographics_file.unlink()
            debug_log(f"Removed demographics file: {demographics_file}")

        # Remove diseases file
        diseases_file = SAVE_DIR / "diseases.json"
        if diseases_file.exists():
            diseases_file.unlink()
            debug_log(f"Removed diseases file: {diseases_file}")

        # Remove all case_* directories
        for item in SAVE_DIR.iterdir():
            if item.is_dir() and item.name.startswith("case_"):
                shutil.rmtree(item)
                debug_log(f"Removed case directory: {item}")

        debug_log("All Python data cleared successfully")

    def _build_enriched_disease(self) -> Dict[str, Any]:
        """
        Build enriched disease object with all fields needed by the UI.
        Called after ML prediction completes to return complete data to frontend.

        Returns:
            Dictionary with all disease fields (id, name, description, bodyPart,
            mapPosition, image, confidenceLevel, llmResponse, timelineData, etc.)
        """
        # Load latest case history and conversation
        case_history = self.case_history or {}

        # Get name from case_history
        name = case_history.get("name", "Unknown Condition")

        # Get earliest date entry for description and confidence
        dates = case_history.get("dates", {})
        earliest_date = min(dates.keys()) if dates else None

        description = ""
        confidenceLevel = 0
        image_path = ""
        if earliest_date:
            entry = dates[earliest_date]
            # Get description from text_summary
            text_summary = entry.get("text_summary", "")
            description = text_summary[:100] + "..." if len(text_summary) > 100 else text_summary
            # Get confidence from top prediction
            predictions = entry.get("predictions", {})
            if predictions:
                confidenceLevel = int(max(predictions.values()) * 100)
            # Get image path
            image_path = entry.get("image_path", "")

        # Get body part from location
        location = case_history.get("location", {})
        bodyPart = location.get("nlp", "Unknown")
        coordinates = location.get("coordinates", None)
        mapPosition = None
        if coordinates and len(coordinates) == 2:
            mapPosition = {"leftPct": coordinates[0], "topPct": coordinates[1]}

        # Get LLM response from conversation history (first LLM message)
        llmResponse = ""
        if self.conversation_history and len(self.conversation_history) > 0:
            first_entry = self.conversation_history[0]
            llmResponse = first_entry.get("llm", {}).get("message", "")

        # Read image as base64 for thumbnail (if it exists)
        image_base64 = ""
        if image_path and Path(image_path).exists():
            try:
                import base64

                with open(image_path, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                debug_log(f"Warning: Could not read image for thumbnail: {e}")

        # Build enriched disease object
        clean_id = self.case_id.replace("case_", "") if self.case_id.startswith("case_") else self.case_id
        enriched_disease = {
            "id": clean_id,
            "name": name,
            "description": description,
            "bodyPart": bodyPart,
            "mapPosition": mapPosition,
            "image": image_base64,
            "confidenceLevel": confidenceLevel,
            "date": earliest_date or "",
            "llmResponse": llmResponse,
            "timelineData": (
                [
                    {
                        "id": i,
                        "date": date,
                        "image": dates[date].get("image_path", ""),
                        "note": dates[date].get("text_summary", ""),
                    }
                    for i, date in enumerate(sorted(dates.keys(), reverse=True))
                ]
                if dates
                else []
            ),
            "conversationHistory": self.conversation_history or [],
        }

        return enriched_disease

    def get_initial_prediction(
        self,
        image_path: str,
        text_description: str,
        user_timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process initial image and text input to generate predictions and LLM analysis.

        Workflow:
        1. Load image from provided path as PIL Image
        2. Run local ML model for embeddings and CV analysis
        3. Send embedding to cloud for disease predictions
        4. Build metadata combining all inputs and history
        5. Call LLM API to generate explanation
        6. Save results for future reference

        Args:
            image_path: Path to image file on disk
            text_description: User's description of their skin condition
            user_timestamp: Optional ISO timestamp of when user initiated request

        Returns:
            Dictionary containing:
                - llm_response: Text explanation from LLM
                - predictions: Disease predictions with confidence scores
                - cv_analysis: Computer vision analysis results
                - embedding: Image embedding vector
        """
        debug_log(f"Processing initial prediction for case {self.case_id}...")

        # Step 0: Load image as PIL and save to case directory
        debug_log("  → Loading and saving image...")
        from PIL import Image

        image = Image.open(image_path)
        saved_image_path = self._save_image(image)

        # Step 1: Run local ML model for embeddings TODO
        debug_log("  → Running local ML model for embeddings...")
        embedding = self._run_local_ml_model(saved_image_path)

        # Step 2: Run CV analysis TODO
        debug_log("  → Running CV analysis...")
        cv_analysis = self._run_cv_analysis(saved_image_path)

        # Step 3: Get predictions from cloud ML model TODO
        debug_log("  → Getting cloud predictions...")
        updated_text_description = self.update_text_input(text_description)
        predictions_raw = self._run_cloud_ml_model(embedding, updated_text_description)

        # Step 3.5: Reformat predictions for explain LLM
        predictions = {item["class"]: item["probability"] for item in predictions_raw}

        # Step 4: Build metadata for LLM
        metadata = {
            "user_input": updated_text_description,
            "cv_analysis": cv_analysis,
            "history": self.case_history["dates"],
        }

        # Step 5: Call LLM API for explanation
        debug_log("  → Calling LLM for explanation...")
        llm_response, llm_timestamp = self._call_llm_explain(predictions, metadata)

        # Step 6: Save to conversation history (initial LLM response)
        debug_log("  → Saving conversation...")
        self._save_conversation_entry(
            user_message=updated_text_description,
            llm_response=llm_response,
            user_timestamp=user_timestamp,
            llm_timestamp=llm_timestamp,
        )

        # Step 7: Save to history (CV analysis, predictions, image path)
        debug_log("  → Saving history...")
        current_date = datetime.now().strftime("%Y-%m-%d")
        self._save_history_entry(
            date=current_date,
            cv_analysis=cv_analysis,
            predictions=predictions,
            image_path=saved_image_path,
            text_summary=updated_text_description,
        )

        # Step 8: Build enriched disease object for frontend
        enriched_disease = self._build_enriched_disease()

        # Step 9: Prepare complete results (keep original fields for backward compatibility)
        results = {
            "llm_response": llm_response,
            "predictions": predictions,
            "cv_analysis": cv_analysis,
            "embedding": embedding,
            "text_description": text_description,
            "enriched_disease": enriched_disease,  # NEW: Complete disease object with all UI fields
        }

        debug_log(f"✓ Initial prediction complete for case {self.case_id}")
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
                age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
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
            (self.case_history or {}).get("location", {}).get("nlp") if isinstance(self.case_history, dict) else None
        )
        if location_nlp:
            demographic_sentences.append(f"The body location of the affected area is {location_nlp}.")

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

        debug_log(f"Processing chat message for case {self.case_id}...")

        # Step 1: Load conversation history
        debug_log("  → Loading conversation history...")
        conversation_data = self._load_json_file(
            file_path=self.conversation_file, default_value=[], description="conversation history"
        )

        if not conversation_data or len(conversation_data) == 0:
            raise ValueError(
                f"No conversation found for case {self.case_id}. " "Please run get_initial_prediction first."
            )

        # Step 2: Extract initial message and build conversation history
        initial_message = conversation_data[0]
        conversation_history = conversation_data[1:]  # All entries after first

        # Step 3: Call LLM API for follow-up
        debug_log("  → Calling LLM for follow-up answer...")
        response, llm_timestamp = self._call_llm_followup(
            initial_message=initial_message,
            question=user_query,
            conversation_history=conversation_history[-5:],  # Last 5 entries
        )

        # Step 4: Save new conversation entry
        debug_log("  → Saving conversation...")
        self._save_conversation_entry(
            user_message=user_query,
            llm_response=response["answer"],
            user_timestamp=user_timestamp,
            llm_timestamp=llm_timestamp,
        )

        debug_log(f"✓ Chat message processed for case {self.case_id}")
        return {"answer": response["answer"], "conversation_history": response["conversation_history"]}

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
            debug_log(f"Error saving conversation history: {exc}")

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
        Also derives and saves disease name if this is the first entry.
        """

        dates = self.case_history.setdefault("dates", {})
        dates[date] = {
            "cv_analysis": cv_analysis,
            "predictions": predictions,
            "image_path": image_path,
            "text_summary": text_summary or "",
        }

        # If this is the first entry and no name exists, derive from top prediction
        if len(dates) == 1 and "name" not in self.case_history:
            if predictions:
                top_disease = max(predictions.items(), key=lambda x: x[1])[0]
                # Format: remove underscores, capitalize first letter of each word
                formatted_name = " ".join(word.capitalize() for word in top_disease.split("_"))
                self.case_history["name"] = formatted_name
                debug_log(f"Derived disease name from top prediction: {formatted_name}")

        try:
            with open(self.case_history_file, "w") as f:
                json.dump(self.case_history, f, indent=2)
        except OSError as exc:
            debug_log(f"Error saving history: {exc}")

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
            debug_log(f"No {description} file found for case {self.case_id}, using default.")
            return deepcopy(default_value)

        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            debug_log(f"Warning: Could not parse {description} file for case {self.case_id}. Using default.")
            return deepcopy(default_value)

    def _run_local_ml_model(self, image_path: str) -> List[float]:
        """
        Run local ML model for embeddings using trained vision encoder.

        Args:
            image_path: Path to the image file

        Returns:
            Image embedding vector (list of floats)
        """
        try:
            debug_log(f"    Encoding image: {image_path}")

            # Get shared encoder instance
            encoder = self.get_vision_encoder()

            # Encode image
            embedding_array = encoder.encode(image_path)
            embedding_list = embedding_array.tolist()

            debug_log(f"    ✓ Embedding extracted (dim={len(embedding_list)})")
            return embedding_list

        except FileNotFoundError:
            debug_log(f"    ✗ Image file not found: {image_path}")
            raise ValueError(f"Image file not found: {image_path}")
        except Exception as e:
            debug_log(f"    ✗ Error extracting embedding: {e}")
            import traceback

            debug_log(traceback.format_exc())
            raise ValueError(f"Error extracting image embedding: {str(e)}")

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
            debug_log("    [DUMMY MODE] Returning dummy CV analysis...")
            return {
                "area": 8.4,
                "color_profile": {"average_Lab": [67.2, 18.4, 9.3], "redness_index": 0.34, "texture_contrast": 0.12},
                "boundary_irregularity": 0.23,
                "symmetry_score": 0.78,
            }

        # TODO: REAL IMPLEMENTATION
        debug_log("    [TODO] Running CV analysis...")
        return {
            "area": 8.4,
            "color_profile": {"average_Lab": [67.2, 18.4, 9.3], "redness_index": 0.34, "texture_contrast": 0.12},
        }

    def _run_cloud_ml_model(self, embedding: List[float], text_description: str) -> Dict[str, float]:
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
            debug_log("    [DUMMY MODE] Returning dummy predictions...")
            return {
                "eczema": 0.78,
                "contact_dermatitis": 0.15,
                "psoriasis": 0.04,
                "tinea_corporis": 0.02,
                "seborrheic_dermatitis": 0.01,
            }

        # Embed text description using cloud API
        debug_log("Calling cloud ML model...")
        response = requests.post(self.text_embed_url, json={"text": text_description}, timeout=60)
        response.raise_for_status()
        text_embedding = response.json().get("embedding", [0.0] * 512)

        debug_log("Vision Embedding Shape: {}".format(len(embedding)))
        debug_log("Text Embedding Shape: {}".format(len(text_embedding)))

        # get predictions from cloud API
        response = requests.post(
            self.prediction_url,
            json={
                "vision_embedding": embedding,
                "text_embedding": text_embedding,
                "top_k": 5,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("top_k", {})

    def _call_llm_explain(self, predictions: Dict[str, float], metadata: Dict[str, Any]) -> Tuple[Any, str]:
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

        debug_log(" Calling LLM explain API...")
        try:
            response = requests.post(
                self.llm_explain_url, json={"predictions": predictions, "metadata": metadata}, timeout=300
            )
            response.raise_for_status()
            llm_timestamp = _timestamp()
            return response.json(), llm_timestamp  # or response.text depending on API format
        except requests.exceptions.RequestException as e:
            debug_log(f"Error calling LLM API: {e}")
            raise

    def _call_llm_followup(
        self, initial_message: Dict[str, Any], question: str, conversation_history: List[Dict[str, Any]]
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

        debug_log(" Calling LLM followup API...")
        try:
            response = requests.post(
                self.llm_followup_url,
                json={
                    "initial_answer": initial_answer,
                    "question": question,
                    "conversation_history": history_questions,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json(), _timestamp()
        except requests.exceptions.RequestException as e:
            debug_log(f"Error calling LLM API: {e}")
            raise
