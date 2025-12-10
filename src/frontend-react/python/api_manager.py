import json
import os
import sys
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor

import requests
from prediction_texts import get_prediction_text
from model_manager import get_model_path

# Disclaimer appended to all LLM responses
DISCLAIMER = "\n\n**Please note:** I'm just a helpful assistant and can't give you a medical diagnosis. This information is for general knowledge, and a doctor is the best person to give you a proper diagnosis and treatment plan."

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
DEFAULT_LLM_FOLLOWUP_URL = "https://tanushkmr2001--dermatology-llm-27b-dermatologyllm-ask-fo-8013b2.modal.run"
DEFAULT_LLM_EXPLAIN_URL = "https://tanushkmr2001--dermatology-llm-27b-dermatologyllm-explai-0d573f.modal.run"
LLM_TIME_TRACKING_URL = "https://tanushkmr2001--dermatology-llm-27b-dermatologyllm-time-t-f8b7ef.modal.run"


def debug_log(msg: str):
    """Print to stderr so it doesn't interfere with stdout JSON protocol"""
    print(msg, file=sys.stderr, flush=True)


def _convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj


# Import CV analysis module (after debug_log is defined)
try:
    sys.path.insert(0, str(Path(__file__).parent / "cv-analysis"))
    from module import run_cv_analysis as cv_run_analysis

    CV_ANALYSIS_AVAILABLE = True
    debug_log("✓ CV analysis module loaded successfully")
except ImportError as e:
    debug_log(f"⚠ Warning: CV analysis module not available: {e}")
    CV_ANALYSIS_AVAILABLE = False


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
        self.case_history_file = self.case_dir / "case_history.json"
        self.conversation_file = self.case_dir / "conversation_history.json"
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

            # Lazy import to avoid heavy torch load at process start
            from inference_local.vision_encoder import VisionEncoder

            # Ensure model file is available (download if needed)
            try:
                checkpoint_path = get_model_path("vision")
                debug_log(f"Using checkpoint: {checkpoint_path}")
            except Exception as e:
                debug_log(f"✗ Error downloading/locating model: {e}")
                # Fallback: try local path for development
                current_file = Path(__file__).resolve()
                python_dir = current_file.parent
                fallback_path = python_dir / "inference_local" / "test_best.pth"

                if fallback_path.exists():
                    debug_log(f"Using fallback local checkpoint: {fallback_path}")
                    checkpoint_path = fallback_path
                else:
                    error_msg = (
                        f"Model not found. Expected at {fallback_path}\n"
                        f"Please either:\n"
                        f"  1. Set up GitHub release URL in model_manager.py\n"
                        f"  2. Place model at python/inference_local/test_best.pth"
                    )
                    debug_log(f"✗ {error_msg}")
                    raise FileNotFoundError(error_msg)

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
                debug_log("Warning: Could not parse demographics file")
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
            debug_log("Warning: Could not parse diseases file")
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
            predictionText = ""
            if earliest_date:
                entry = dates[earliest_date]
                # Get description from text_summary
                description = entry.get("text_summary", "")[:100] + "..." if entry.get("text_summary", "") else ""
                # Get confidence from top prediction
                predictions = entry.get("predictions", {})
                if predictions:
                    confidenceLevel = int(max(predictions.values()) * 100)
                    # Get top prediction label and look up predefined text
                    top_prediction_label = max(predictions.items(), key=lambda x: x[1])[0]
                    predictionText = get_prediction_text(top_prediction_label)

            # Get body part from location
            location = case_history.get("location", {})
            bodyPart = location.get("nlp", "Unknown")
            coordinates = location.get("coordinates", None)
            mapPosition = None
            if coordinates and len(coordinates) == 2:
                mapPosition = {"leftPct": coordinates[0], "topPct": coordinates[1]}

            # Get LLM response from conversation history (first LLM message)
            # Also get last message timestamp for sorting
            llmResponse = ""
            lastMessageTimestamp = ""
            conversation_file = SAVE_DIR / folder_id / "conversation_history.json"
            if conversation_file.exists():
                try:
                    with open(conversation_file, "r") as f:
                        conversation = json.load(f)
                        # Get first LLM response
                        if conversation and len(conversation) > 0:
                            first_entry = conversation[0]
                            llmResponse = first_entry.get("llm", {}).get("message", "")

                            # Get last message timestamp (from last entry, prefer LLM timestamp, fallback to user)
                            last_entry = conversation[-1]
                            lastMessageTimestamp = (
                                last_entry.get("llm", {}).get("timestamp")
                                or last_entry.get("user", {}).get("timestamp")
                                or ""
                            )
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
                "predictionText": predictionText,  # Predefined text based on prediction
                "lastMessageTimestamp": lastMessageTimestamp,  # Timestamp of last message for sorting
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
    def add_timeline_entry(case_id: str, image_path: str, note: str, date: str, has_coin: bool = False) -> None:
        """
        Add a new timeline entry (manually uploaded image) to case_history.json.
        Conditionally includes CV analysis and time tracking summary generation.

        Args:
            case_id: Case ID (with or without 'case_' prefix)
            image_path: Path to the saved image file
            note: User's notes/description for this entry
            date: ISO date string (YYYY-MM-DD) for when this entry was added
            has_coin: Whether the image contains a coin for size reference
        """
        # Ensure case_id has 'case_' prefix for folder lookup
        if not case_id.startswith("case_"):
            case_id = f"case_{case_id}"

        # Create temporary APIManager instance to use its methods
        api = APIManager(case_id, dummy=False)

        debug_log(f"[api_manager] add_timeline_entry called with has_coin={has_coin} (type: {type(has_coin).__name__})")
        debug_log(f"Adding timeline entry for case {case_id} (has_coin={has_coin})")

        # Get predictions from the earliest date entry (initial diagnosis)
        predictions = {}
        top_prediction_label = ""
        if api.case_history.get("dates"):
            earliest_date = min(api.case_history["dates"].keys())
            earliest_entry = api.case_history["dates"][earliest_date]
            predictions = earliest_entry.get("predictions", {})
            if predictions:
                top_prediction_label = max(predictions.items(), key=lambda x: x[1])[0]

        # Determine if we should run CV analysis
        should_run_cv = has_coin

        cv_analysis = {}
        tracking_summary = ""

        if should_run_cv:
            # Run CV analysis on the uploaded image
            debug_log("  → Running CV analysis...")
            cv_analysis = api._run_cv_analysis(image_path)

            # Generate time tracking summary
            debug_log("  → Generating time tracking summary...")
            tracking_summary = api._get_time_tracking_summary(
                predictions=predictions, text_description=note, cv_analysis=cv_analysis
            )
        else:
            debug_log(f"  → Skipping CV analysis (has_coin={has_coin}, top_prediction={top_prediction_label})")

        # Clean cv_analysis to remove numpy arrays for JSON serialization
        cv_analysis_clean = {}
        if cv_analysis:
            for key, value in cv_analysis.items():
                if key == "masks" or key == "images":
                    # Skip numpy arrays (masks and images)
                    continue
                elif key == "coin_data" and value is not None:
                    # Convert tuple to list for JSON serialization
                    cv_analysis_clean[key] = list(value) if isinstance(value, tuple) else value
                else:
                    cv_analysis_clean[key] = value

        # Convert numpy types (int64, float64, etc.) to Python native types
        cv_analysis_clean = _convert_numpy_types(cv_analysis_clean)

        # Add new entry to dates using the provided date as key
        if "dates" not in api.case_history:
            api.case_history["dates"] = {}

        api.case_history["dates"][date] = {
            "image_path": image_path,
            "text_summary": note or "",
            "cv_analysis": cv_analysis_clean,
            "predictions": predictions,  # Use predictions from initial entry
            "tracking_summary": tracking_summary,
        }

        # Save updated case history
        APIManager.save_case_history(case_id, api.case_history)
        debug_log(f"✅ Added timeline entry for case {case_id} on date {date}")

    @staticmethod
    def delete_cases(case_ids: List[str]) -> None:
        """
        Delete specified cases by removing their folders and updating diseases.json.

        Args:
            case_ids: List of case IDs (with or without 'case_' prefix) to delete
        """
        diseases_file = SAVE_DIR / "diseases.json"

        # Load current diseases list
        diseases = []
        if diseases_file.exists():
            try:
                with open(diseases_file, "r") as f:
                    diseases = json.load(f)
            except json.JSONDecodeError:
                debug_log("Warning: Could not parse diseases file")
                diseases = []

        # Normalize case IDs (ensure they have 'case_' prefix for folder lookup)
        normalized_case_ids = []
        for case_id in case_ids:
            if case_id.startswith("case_"):
                normalized_case_ids.append(case_id)
            else:
                normalized_case_ids.append(f"case_{case_id}")

        # Delete case folders
        deleted_count = 0
        for case_id in normalized_case_ids:
            case_dir = SAVE_DIR / case_id
            if case_dir.exists() and case_dir.is_dir():
                shutil.rmtree(case_dir)
                debug_log(f"Deleted case directory: {case_dir}")
                deleted_count += 1
            else:
                debug_log(f"Warning: Case directory not found: {case_dir}")

        # Remove deleted cases from diseases list
        # Extract clean IDs (without 'case_' prefix) for comparison
        clean_case_ids = {cid.replace("case_", "") if cid.startswith("case_") else cid for cid in case_ids}
        diseases = [d for d in diseases if d.get("id") not in clean_case_ids]

        # Save updated diseases list
        with open(diseases_file, "w") as f:
            json.dump(diseases, f, indent=2)

        debug_log(f"Deleted {deleted_count} case(s) and updated diseases.json")

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
        predictionText = ""
        if earliest_date:
            entry = dates[earliest_date]
            # Get description from text_summary
            text_summary = entry.get("text_summary", "")
            description = text_summary[:100] + "..." if len(text_summary) > 100 else text_summary
            # Get confidence from top prediction
            predictions = entry.get("predictions", {})
            if predictions:
                confidenceLevel = int(max(predictions.values()) * 100)
                # Get top prediction label and look up predefined text
                top_prediction_label = max(predictions.items(), key=lambda x: x[1])[0]
                predictionText = get_prediction_text(top_prediction_label)
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
        # Also get last message timestamp for sorting
        llmResponse = ""
        lastMessageTimestamp = ""
        if self.conversation_history and len(self.conversation_history) > 0:
            first_entry = self.conversation_history[0]
            llmResponse = first_entry.get("llm", {}).get("message", "")

            # Get last message timestamp (from last entry, prefer LLM timestamp, fallback to user)
            last_entry = self.conversation_history[-1]
            lastMessageTimestamp = (
                last_entry.get("llm", {}).get("timestamp") or last_entry.get("user", {}).get("timestamp") or ""
            )

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
            "predictionText": predictionText,
            "lastMessageTimestamp": lastMessageTimestamp,  # Timestamp of last message for sorting
            "timelineData": (
                [
                    {
                        "id": i,
                        "date": date,
                        "image": dates[date].get("image_path", ""),
                        "note": dates[date].get("text_summary", ""),
                        "trackingSummary": dates[date].get("tracking_summary", ""),
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
        on_chunk: Optional[Callable[[str], None]] = None,
        has_coin: bool = False,
    ) -> Dict[str, Any]:
        """
        Process initial image and text input to generate predictions and LLM analysis.

        Workflow:
        1. Load image from provided path as PIL Image
        2. Run local ML model for embeddings and CV analysis
        3. Send embedding to cloud for disease predictions
        4. Build metadata combining all inputs and history
        5. Call LLM API to generate explanation (streaming if on_chunk provided)
        6. Save results for future reference

        Args:
            image_path: Path to image file on disk
            text_description: User's description of their skin condition
            user_timestamp: Optional ISO timestamp of when user initiated request
            on_chunk: Optional callback function for streaming LLM explanation chunks

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

        # Normalize and validate image path
        image_path = str(Path(image_path).resolve())
        debug_log(f"    Image path: {image_path}")
        debug_log(f"    Path exists: {Path(image_path).exists()}")

        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found at path: {image_path}")

        image = Image.open(image_path)
        saved_image_path = self._save_image(image)

        # Verify saved image path exists before using it
        if saved_image_path and not Path(saved_image_path).exists():
            raise FileNotFoundError(f"Saved image not found at: {saved_image_path}")

        # Step 1: Run CV analysis first if needed, then remove coin from image
        cv_analysis = {}
        processed_image = image
        debug_log(f"  → has_coin={has_coin}")
        if has_coin:
            debug_log("  → Running CV analysis to detect coin...")
            cv_result = self._run_cv_analysis(saved_image_path)
            cv_analysis = cv_result

            # Remove coin from image if detected
            coin_mask_full = cv_result.get("masks", {}).get("coin_mask_full")
            debug_log(
                f"  → Coin mask check: mask is None={coin_mask_full is None}, has_data={coin_mask_full is not None and coin_mask_full.any() if coin_mask_full is not None else False}"
            )

            if coin_mask_full is not None and coin_mask_full.any():
                debug_log("  → Cropping coin from image...")
                import cv2
                import numpy as np

                # Convert PIL image to OpenCV format
                img_array = np.array(image.convert("RGB"))
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # Ensure image is 8-bit 3-channel BGR
                if img_bgr.dtype != np.uint8:
                    img_bgr = img_bgr.astype(np.uint8)
                if len(img_bgr.shape) == 2:
                    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
                elif img_bgr.shape[2] == 4:
                    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

                # Resize coin mask to match image if needed (CV analysis may downscale)
                h, w = img_bgr.shape[:2]
                if coin_mask_full.shape[:2] != (h, w):
                    coin_mask_full = cv2.resize(coin_mask_full, (w, h), interpolation=cv2.INTER_NEAREST)

                # Ensure mask is 8-bit 1-channel
                if coin_mask_full.dtype != np.uint8:
                    coin_mask_full = coin_mask_full.astype(np.uint8)
                if len(coin_mask_full.shape) > 2:
                    coin_mask_full = cv2.cvtColor(coin_mask_full, cv2.COLOR_BGR2GRAY)

                # Find bounding box of coin
                coin_coords = np.where(coin_mask_full > 0)
                if len(coin_coords[0]) > 0:
                    coin_y_min, coin_y_max = coin_coords[0].min(), coin_coords[0].max()
                    coin_x_min, coin_x_max = coin_coords[1].min(), coin_coords[1].max()

                    # Add padding to coin bounding box to ensure full removal
                    padding = 15
                    coin_y_min = max(0, coin_y_min - padding)
                    coin_y_max = min(h, coin_y_max + padding)
                    coin_x_min = max(0, coin_x_min - padding)
                    coin_x_max = min(w, coin_x_max + padding)

                    # Calculate distances from coin to each edge
                    dist_left = coin_x_min
                    dist_right = w - coin_x_max
                    dist_top = coin_y_min
                    dist_bottom = h - coin_y_max

                    # Find the edge closest to the coin (coin is likely near that edge)
                    min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

                    # Crop from the edge closest to the coin
                    if min_dist == dist_left:
                        # Coin is on left, crop from left (keep right side)
                        crop_x = coin_x_max
                        crop_y = 0
                        crop_w = w - coin_x_max
                        crop_h = h
                        debug_log(f"  → Coin on left edge, cropping from x={crop_x}, keeping {crop_w}x{crop_h}")
                    elif min_dist == dist_right:
                        # Coin is on right, crop from right (keep left side)
                        crop_x = 0
                        crop_y = 0
                        crop_w = coin_x_min
                        crop_h = h
                        debug_log(f"  → Coin on right edge, cropping to x={crop_w}, keeping {crop_w}x{crop_h}")
                    elif min_dist == dist_top:
                        # Coin is on top, crop from top (keep bottom side)
                        crop_x = 0
                        crop_y = coin_y_max
                        crop_w = w
                        crop_h = h - coin_y_max
                        debug_log(f"  → Coin on top edge, cropping from y={crop_y}, keeping {crop_w}x{crop_h}")
                    else:  # dist_bottom
                        # Coin is on bottom, crop from bottom (keep top side)
                        crop_x = 0
                        crop_y = 0
                        crop_w = w
                        crop_h = coin_y_min
                        debug_log(f"  → Coin on bottom edge, cropping to y={crop_h}, keeping {crop_w}x{crop_h}")

                    # Ensure we have a valid crop (at least some pixels)
                    if crop_w > 0 and crop_h > 0:
                        # Perform the crop
                        img_bgr = img_bgr[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
                        debug_log(f"  → ✅ Cropped image to size: {img_bgr.shape[1]}x{img_bgr.shape[0]}")
                    else:
                        debug_log("  → ⚠️ Invalid crop dimensions, skipping crop")
                else:
                    debug_log("  → ⚠️ Could not find coin coordinates in mask")

                # Convert back to PIL Image
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                processed_image = Image.fromarray(img_rgb)

            else:
                debug_log("  → ⚠️ No coin detected or coin mask is empty, skipping coin removal")

        # Step 2: Run local ML model for embeddings (using coin-removed image)
        debug_log("  → Running local ML model for embeddings...")
        embedding = self._run_local_ml_model(processed_image)

        # Step 3: Get predictions from cloud ML model
        debug_log("  → Getting cloud predictions...")
        updated_text_description = self.update_text_input(text_description)
        predictions_raw = self._run_cloud_ml_model(embedding, updated_text_description)

        # Step 2.5: Reformat predictions for explain LLM
        predictions = {item["class"]: item["probability"] for item in predictions_raw}

        # Step 3: Generate time tracking summary from CV + history (for both tracking panel and explain prompt)
        debug_log("  → Generating time tracking summary...")
        tracking_summary = self._get_time_tracking_summary(
            predictions=predictions,
            text_description=text_description,  # ORIGINAL user text
            cv_analysis=cv_analysis,
        )

        # Step 4: Build metadata for LLM (include CV tracking summary text, not raw metrics)
        metadata = {
            "user_input": updated_text_description,
            "history": self.case_history["dates"],
            "cv_tracking_summary": tracking_summary,
        }

        # Step 5: Call LLM API for explanation
        debug_log("  → Calling LLM for explanation...")
        llm_response_dict, llm_timestamp = self._call_llm_explain(
            predictions=predictions,
            metadata=metadata,
            on_chunk=on_chunk if on_chunk else (lambda x: None),  # Use provided callback or no-op
        )
        # Extract the answer string from the response dict (for backward compatibility)
        llm_response = (
            llm_response_dict.get("answer", "") if isinstance(llm_response_dict, dict) else str(llm_response_dict)
        )
        # Append disclaimer to the response
        llm_response = llm_response + DISCLAIMER

        # Step 6: Save to conversation history (initial LLM response)
        # Save ORIGINAL user text (without demographics) to conversation
        debug_log("  → Saving conversation...")
        self._save_conversation_entry(
            user_message=text_description,  # Use original text, not augmented
            llm_response=llm_response,
            user_timestamp=user_timestamp,
            llm_timestamp=llm_timestamp,
        )

        # Step 7: Save to history (CV analysis, predictions, image path, tracking summary)
        # Use ORIGINAL text_description for text_summary, not the augmented version
        debug_log("  → Saving history...")
        current_date = datetime.now().strftime("%Y-%m-%d")
        self._save_history_entry(
            date=current_date,
            cv_analysis=cv_analysis,
            predictions=predictions,
            image_path=saved_image_path,
            text_summary=text_description,  # Save ORIGINAL user input, not augmented
            tracking_summary=tracking_summary,
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

    from typing import Optional, Callable, Dict, Any

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
        tracking_summary: Optional[str] = None,
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
            "tracking_summary": tracking_summary or "",
        }

        # Log what we're saving for debugging
        debug_log(f"    Saved to history[{date}]:")
        debug_log(f"      - text_summary: {(text_summary or '')[:80]}...")
        debug_log(f"      - tracking_summary: {(tracking_summary or '')[:80]}...")
        debug_log(f"      - cv_analysis keys: {list(cv_analysis.keys()) if cv_analysis else 'None'}")

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
            # Save the image
            image.save(destination)

            # Verify the file was actually saved
            destination_path = Path(destination).resolve()
            if not destination_path.exists():
                raise FileNotFoundError(f"Image save failed: file not found at {destination_path}")

            debug_log(f"    ✓ Image saved to: {destination_path}")
            return str(destination_path)

        except Exception as exc:
            debug_log(f"    ✗ Error saving image: {exc}")
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

        except Exception as e:
            debug_log(f"    ✗ Error extracting embedding: {e}")
            import traceback

            debug_log(traceback.format_exc())
            raise ValueError(f"Error extracting image embedding: {str(e)}")

    def _run_cv_analysis(self, image_path: str) -> Dict[str, Any]:
        """
        Run computer vision analysis on the image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing CV analysis metrics:
                - compactness_index: Shape metric (circularity)
                - color_stats_lab: LAB color space statistics
                - area_cm2: Lesion area in cm² (if coin detected)
                - tilt_correction_factor: Correction factor for tilted coins
        """
        if self.dummy:
            debug_log("    [DUMMY MODE] Returning dummy CV analysis...")
            return {
                "compactness_index": 1.2,
                "color_stats_lab": {
                    "mean_L": 67.2,
                    "mean_A": 18.4,
                    "mean_B": 9.3,
                    "std_L": 12.1,
                    "std_A": 5.3,
                    "std_B": 4.2,
                },
                "area_cm2": 8.4,
                "tilt_correction_factor": 1.0,
            }

        # Use actual CV analysis if available
        if CV_ANALYSIS_AVAILABLE:
            try:
                debug_log(f"    Running CV analysis on: {image_path}")
                cv_results = cv_run_analysis(image_path, bbox=None)

                # Return full result including masks (needed for coin removal)
                # Flatten metrics at top level for backward compatibility with existing code
                metrics = cv_results.get("metrics", {})
                result = dict(metrics)  # Start with flattened metrics
                result["masks"] = cv_results.get("masks", {})
                result["coin_data"] = cv_results.get("coin_data")

                debug_log(f"    CV analysis completed. Metrics: {json.dumps(metrics, indent=2)}")
                return result

            except Exception as e:
                debug_log(f"    ⚠ CV analysis failed: {e}")
                import traceback

                debug_log(traceback.format_exc())
                # Fall through to dummy data

        # Fallback to dummy data if CV analysis not available or failed
        debug_log("    [FALLBACK] Using dummy CV analysis...")
        return {
            "compactness_index": 1.2,
            "color_stats_lab": {
                "mean_L": 67.2,
                "mean_A": 18.4,
                "mean_B": 9.3,
                "std_L": 12.1,
                "std_A": 5.3,
                "std_B": 4.2,
            },
            "area_cm2": None,  # No coin detected
            "tilt_correction_factor": None,
            "masks": {},  # No masks in dummy mode
            "coin_data": None,
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
        result = response.json()

        # Get top_k predictions (even if uncertain)
        # Handle both list and dict formats
        top_k_raw = result.get("top_k", {})

        # Convert list format to dict if needed
        if isinstance(top_k_raw, list):
            top_k_predictions = {
                item.get("class", item.get("disease", "")): item.get("probability", item.get("prob", 0.0))
                for item in top_k_raw
            }
        else:
            top_k_predictions = top_k_raw if isinstance(top_k_raw, dict) else {}

        # Check if model is uncertain about the prediction
        if result.get("predicted_class") == "UNCERTAIN":
            debug_log("⚠️ Model returned UNCERTAIN prediction")
            # Still log the top k predictions for debugging
            if top_k_predictions:
                debug_log("Top K predictions (even though uncertain):")
                for disease, prob in sorted(top_k_predictions.items(), key=lambda x: x[1], reverse=True):
                    debug_log(f"  - {disease}: {prob:.4f}")
            return {"UNCERTAIN": 1.0}

        return top_k_predictions

    def _call_llm_explain(
        self,
        predictions: Dict[str, float],
        metadata: Dict[str, Any],
        on_chunk: Callable[[str], None],  # streaming callback required
    ) -> Tuple[Dict[str, Any], str]:
        """
        Call the LLM explanation API in FULL STREAMING MODE.

        Streams incremental explanation text via on_chunk(),
        assembles the final response, and returns (response_dict, timestamp).
        """

        debug_log("Calling LLM explain API (streaming)...")

        payload = {
            "predictions": predictions,
            "metadata": metadata,
        }

        # Log the full prompt/payload being sent to the LLM
        debug_log("\n" + "=" * 80)
        debug_log("📤 FULL PROMPT SENT TO LLM EXPLAIN API:")
        debug_log("=" * 80)
        debug_log(json.dumps(payload, indent=2, default=str))
        debug_log("=" * 80 + "\n")

        try:
            full_answer_parts: List[str] = []
            final_response: Optional[Dict[str, Any]] = None

            with requests.post(
                self.llm_explain_url,  # IMPORTANT: use streaming URL
                json=payload,
                stream=True,
                timeout=600,
            ) as resp:
                resp.raise_for_status()

                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue

                    chunk = None

                    # Try JSON-structured chunk first (preferred)
                    try:
                        data = json.loads(raw_line)

                        # If server yields {"delta": "..."} or similar
                        if isinstance(data, dict):
                            chunk = data.get("delta") or data.get("text") or data.get("answer_chunk") or None

                            # If server ever sends a final structured block (optional)
                            if "answer" in data:
                                final_response = data

                    except json.JSONDecodeError:
                        # Fallback: raw text chunk
                        chunk = raw_line

                    # Deliver streamed chunk to upper layers
                    if chunk:
                        full_answer_parts.append(chunk)
                        try:
                            on_chunk(chunk)
                        except Exception as cb_err:
                            debug_log(f"on_chunk callback error in explain: {cb_err}")

            # Build final answer text
            full_answer = "".join(full_answer_parts).strip()

            # If LLM server provided final structured answer, prefer it
            if final_response:
                if "answer" not in final_response:
                    final_response["answer"] = full_answer
                return final_response, _timestamp()

            # Otherwise synthesize a structured response
            return {
                "answer": full_answer,
                "predictions": predictions,
                "metadata": metadata,
            }, _timestamp()

        except requests.exceptions.RequestException as e:
            debug_log(f"Error calling LLM explain API (streaming): {e}")
            raise

    def _call_llm_followup(
        self,
        initial_message: Dict[str, Any],
        question: str,
        conversation_history: List[Dict[str, Any]],
        on_chunk: Callable[[str], None],  # required now, not optional
    ) -> Tuple[Dict[str, Any], str]:
        """
        Call LLM API follow-up in fully streaming mode.
        The endpoint MUST stream newline-delimited JSON messages.
        """

        initial_answer = initial_message.get("llm", {}).get("message", "")

        history_questions = [
            entry.get("user", {}).get("message", "")
            for entry in conversation_history
            if entry.get("user", {}).get("message")
        ]

        payload = {
            "initial_answer": initial_answer,
            "question": question,
            "conversation_history": history_questions,
        }

        # Log the full prompt/payload being sent to the LLM
        debug_log("\n" + "=" * 80)
        debug_log("📤 FULL PROMPT SENT TO LLM FOLLOWUP API:")
        debug_log("=" * 80)
        debug_log(json.dumps(payload, indent=2, default=str))
        debug_log("=" * 80 + "\n")

        debug_log("Calling LLM followup API (streaming only)...")

        try:
            full_answer_parts: List[str] = []
            final_response: Optional[Dict[str, Any]] = None

            # Always stream
            with requests.post(
                self.llm_followup_url,
                json=payload,
                stream=True,
                timeout=300,
            ) as resp:
                resp.raise_for_status()

                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue

                    # Try JSON; fallback to raw text
                    chunk = None
                    try:
                        data = json.loads(raw_line)
                        # Handle case where endpoint sends JSON-encoded strings
                        if isinstance(data, str):
                            chunk = data
                        elif isinstance(data, dict):
                            # Extract chunk from dict format
                            chunk = data.get("delta") or data.get("text") or data.get("answer_chunk") or None
                            # If this is the *final* JSON package:
                            if "answer" in data:
                                final_response = data
                    except json.JSONDecodeError:
                        # If not JSON, treat as plain text
                        chunk = raw_line

                    # Send incremental chunk to UI
                    if chunk:
                        full_answer_parts.append(chunk)
                        try:
                            on_chunk(chunk)
                        except Exception as cb_err:
                            debug_log(f"on_chunk callback error: {cb_err}")

            # Build final content
            full_answer = "".join(full_answer_parts).strip()

            # If server provided a structured final answer JSON, use that
            if final_response:
                if "answer" not in final_response:
                    final_response["answer"] = full_answer
                if "conversation_history" not in final_response:
                    final_response["conversation_history"] = conversation_history
                return final_response, _timestamp()

            # Otherwise synthesize a final response
            return {
                "answer": full_answer,
                "conversation_history": conversation_history,
            }, _timestamp()

        except requests.exceptions.RequestException as e:
            debug_log(f"Error calling LLM API (streaming): {e}")
            raise

    def _get_time_tracking_summary(
        self,
        predictions: Dict[str, float],
        text_description: str,
        cv_analysis: Dict[str, Any],
    ) -> str:
        """
        Generate a brief tracking summary from CV analysis and predictions.

        Args:
            predictions: Disease predictions from ML model
            text_description: User's text description (for first entry) or note (for subsequent entries)
            cv_analysis: Current CV metrics

        Returns:
            String summary (2 sentences)
        """
        if self.dummy:
            debug_log("    [DUMMY MODE] Returning dummy tracking summary...")
            return "The affected area measures roughly 2.5 cm² with moderate color variation. The shape appears fairly regular."

        debug_log("Calling LLM time tracking summary API...")

        # Build CV analysis history from case history
        cv_analysis_history = {}
        for date, entry in self.case_history.get("dates", {}).items():
            if "cv_analysis" in entry:
                # Convert numpy types in existing history entries
                cv_analysis_history[date] = _convert_numpy_types(entry["cv_analysis"])

        # Add current analysis (exclude numpy arrays/masks for JSON serialization)
        current_date = datetime.now().strftime("%Y-%m-%d")
        # Create a JSON-serializable version of cv_analysis (exclude masks and images)
        cv_analysis_serializable = {}
        for key, value in cv_analysis.items():
            if key == "masks" or key == "images":
                # Skip numpy arrays (masks and images)
                continue
            elif key == "coin_data" and value is not None:
                # Convert tuple to list for JSON serialization
                cv_analysis_serializable[key] = list(value) if isinstance(value, tuple) else value
            else:
                cv_analysis_serializable[key] = value

        # Convert numpy types in the serializable version
        cv_analysis_history[current_date] = _convert_numpy_types(cv_analysis_serializable)

        # Determine if this is the first entry
        is_first_entry = len(cv_analysis_history) == 1

        # For first entry, augment user text with demographics
        # For subsequent entries, just use the note as-is
        if is_first_entry:
            user_input_with_context = self.update_text_input(text_description)
        else:
            user_input_with_context = text_description

        payload = {
            "user_input": user_input_with_context,
            "cv_analysis_history": cv_analysis_history,
        }

        # Log the full payload
        debug_log("\n" + "=" * 80)
        debug_log("📤 TIME TRACKING SUMMARY REQUEST:")
        debug_log("=" * 80)
        debug_log(json.dumps(payload, indent=2, default=str))
        debug_log("=" * 80 + "\n")

        try:
            # Call the time tracking summary endpoint
            time_tracking_url = LLM_TIME_TRACKING_URL

            response = requests.post(
                time_tracking_url,
                json=payload,
                timeout=400,
            )
            response.raise_for_status()
            result = response.json()

            summary = result.get("summary", "") or ""

            # Guard against empty/missing summaries – synthesize a simple one from CV metrics
            if not summary.strip():
                debug_log("⚠ LLM returned empty tracking summary; generating fallback from CV metrics")
                area = cv_analysis.get("area_cm2") or cv_analysis.get("area_cm2_uncorrected")
                compactness = cv_analysis.get("compactness_index")
                color = cv_analysis.get("color_stats_lab") or {}
                mean_L = color.get("mean_L")
                mean_A = color.get("mean_A")

                sentences = []
                if area:
                    sentences.append(f"The affected area measures roughly {area:.1f} cm² based on the latest image.")
                if mean_A is not None:
                    sentences.append(
                        "The redness level looks moderate and fairly consistent across the lesion."
                        if mean_A < 25
                        else "The redness level appears relatively high and should be monitored over time."
                    )
                if compactness:
                    sentences.append(
                        "The shape is reasonably regular for this type of spot."
                        if compactness < 3.0
                        else "The shape is a bit irregular, so changes in the edges should be watched."
                    )
                if not sentences:
                    sentences.append(
                        "We have recorded this image for tracking. Future images will help show whether the spot is getting larger, smaller, or staying stable."
                    )

                summary = " ".join(sentences[:3])

            debug_log(f"✅ Time tracking summary received (or synthesized): {summary[:100]}...")
            return summary

        except requests.exceptions.RequestException as e:
            debug_log(f"⚠ Error calling time tracking summary API: {e}")
            # Return a generic fallback
            return (
                "We have recorded this image for tracking, but were unable to generate a detailed summary right now. "
                "Future images will help show whether the spot is changing over time."
            )

    def chat_message(
        self,
        user_query: str,
        on_chunk: Callable[[str], None],
        user_timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle follow-up chat messages from the user.
        """

        debug_log(f"Processing chat message for case {self.case_id}...")

        # Step 1: Load conversation history
        debug_log("  → Loading conversation history...")
        conversation_data = self._load_json_file(
            file_path=self.conversation_file,
            default_value=[],
            description="conversation history",
        )

        if not conversation_data or len(conversation_data) == 0:
            raise ValueError(
                f"No conversation found for case {self.case_id}. " "Please run get_initial_prediction first."
            )

        # Step 2: Extract initial message and build conversation history
        initial_message = conversation_data[0]
        conversation_history = conversation_data[1:]

        # Step 3: Call LLM API for follow-up (streaming if on_chunk is provided)
        debug_log("  → Calling LLM for follow-up answer...")
        response, llm_timestamp = self._call_llm_followup(
            initial_message=initial_message,
            question=user_query,
            conversation_history=conversation_history[-5:],  # Last 5
            on_chunk=on_chunk,  # ← IMPORTANT
        )

        # Append disclaimer to the response
        response["answer"] = response["answer"] + DISCLAIMER

        # Step 4: Save new conversation entry
        debug_log("  → Saving conversation...")
        self._save_conversation_entry(
            user_message=user_query,
            llm_response=response["answer"],
            user_timestamp=user_timestamp,
            llm_timestamp=llm_timestamp,
        )

        debug_log(f"✓ Chat message processed for case {self.case_id}")
        return {
            "answer": response["answer"],
            "conversation_history": response.get("conversation_history", []),
        }
