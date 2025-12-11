#!/usr/bin/env python3
"""
Rebuild diseases.json from existing case folders.
"""

import json
import base64
from pathlib import Path

# Use absolute path
PYTHON_DIR = Path(__file__).parent


def get_first_image_thumbnail(case_dir):
    """Get thumbnail from first image in case folder."""
    images_dir = case_dir / "images"
    if not images_dir.exists():
        return None

    # Find first image file
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    if not image_files:
        return None

    # Read first image and convert to base64
    try:
        with open(image_files[0], "rb") as f:
            image_data = f.read()
            return f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
    except Exception as e:
        print(f"Error reading image {image_files[0]}: {e}")
        return None


def format_name_from_prediction(pred_label):
    """Format disease name from prediction label."""
    # Remove underscores and capitalize each word
    return " ".join(word.capitalize() for word in pred_label.split("_"))


def rebuild_diseases():
    """Rebuild diseases.json from existing case folders."""
    diseases = []

    # Find all case_* folders
    case_folders = [f for f in PYTHON_DIR.iterdir() if f.is_dir() and f.name.startswith("case_")]

    print(f"Found {len(case_folders)} case folders")

    for case_folder in case_folders:
        case_id = case_folder.name.replace("case_", "")
        case_history_path = case_folder / "case_history.json"

        if not case_history_path.exists():
            print(f"Warning: No case_history.json found for {case_folder.name}, skipping")
            continue

        # Load case history
        with open(case_history_path, "r") as f:
            case_history = json.load(f)

        # Get name from case_history or derive from top prediction
        name = case_history.get("name")

        if not name:
            # Try to get from earliest date's top prediction
            dates = case_history.get("dates", {})
            if dates:
                earliest_date = sorted(dates.keys())[0]
                predictions = dates[earliest_date].get("predictions", {})
                if predictions:
                    top_prediction = max(predictions.items(), key=lambda x: x[1])[0]
                    name = format_name_from_prediction(top_prediction)
                    print(f"Derived name '{name}' from top prediction for {case_folder.name}")

                    # Save name to case_history
                    case_history["name"] = name
                    with open(case_history_path, "w") as f:
                        json.dump(case_history, f, indent=2)

        # Get thumbnail image
        thumbnail = get_first_image_thumbnail(case_folder)

        if not thumbnail:
            print(f"Warning: No thumbnail found for {case_folder.name}")

        # Create disease entry (minimal: id, name, image)
        disease = {"id": case_id, "name": name or f"Case {case_id}", "image": thumbnail or ""}

        diseases.append(disease)
        print(f"Added disease: {disease['name']} (ID: {case_id})")

    # Save diseases.json
    diseases_path = PYTHON_DIR / "diseases.json"
    with open(diseases_path, "w") as f:
        json.dump(diseases, f, indent=2)

    print(f"\nRebuilt diseases.json with {len(diseases)} entries")


if __name__ == "__main__":
    rebuild_diseases()
