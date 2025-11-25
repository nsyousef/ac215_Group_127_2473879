#!/usr/bin/env python3
"""
Migration script to update data structure from old format to new simplified format.

Old format:
- diseases.json: Contains full disease objects with description, bodyPart, mapPosition, timeline, etc.
- case_history.json: Only contains dates and location

New format:
- diseases.json: Only contains id, name, and image (thumbnail)
- case_history.json: Contains dates, location, and name (derived from top prediction)
"""
import json
from pathlib import Path

SAVE_DIR = Path(__file__).parent


def migrate():
    print("Starting data migration...")
    
    # 1. Load old diseases.json
    diseases_file = SAVE_DIR / "diseases.json"
    if not diseases_file.exists():
        print("No diseases.json found. Nothing to migrate.")
        return
    
    try:
        with open(diseases_file, 'r') as f:
            old_diseases = json.load(f)
    except json.JSONDecodeError:
        print("Error: Could not parse diseases.json")
        return
    
    print(f"Found {len(old_diseases)} diseases to migrate")
    
    # 2. For each disease, update case_history.json and simplify diseases.json entry
    new_diseases = []
    
    for disease in old_diseases:
        disease_id = disease.get('id')
        if not disease_id:
            print(f"Warning: Skipping disease without ID")
            continue
        
        case_id = f"case_{disease_id}"
        case_dir = SAVE_DIR / case_id
        case_history_file = case_dir / "case_history.json"
        
        # Load case history
        if not case_history_file.exists():
            print(f"Warning: No case_history.json found for {case_id}, skipping")
            continue
        
        try:
            with open(case_history_file, 'r') as f:
                case_history = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not parse case_history.json for {case_id}")
            continue
        
        # Add 'name' field to case_history if it doesn't exist
        if 'name' not in case_history:
            # Try to get name from disease object, or derive from top prediction
            name = disease.get('name')
            if not name:
                # Derive from earliest date's top prediction
                dates = case_history.get('dates', {})
                if dates:
                    earliest_date = min(dates.keys())
                    predictions = dates[earliest_date].get('predictions', {})
                    if predictions:
                        top_disease = max(predictions.items(), key=lambda x: x[1])[0]
                        name = ' '.join(word.capitalize() for word in top_disease.split('_'))
                    else:
                        name = "Unknown Condition"
                else:
                    name = "Unknown Condition"
            
            case_history['name'] = name
            
            # Save updated case_history
            with open(case_history_file, 'w') as f:
                json.dump(case_history, f, indent=2)
            
            print(f"✓ Added 'name' field to {case_id}: {name}")
        else:
            print(f"  {case_id} already has 'name' field: {case_history['name']}")
        
        # Create simplified disease entry
        new_disease = {
            'id': disease_id,
            'name': case_history.get('name', disease.get('name', 'Unknown Condition')),
            'image': disease.get('image')  # Keep thumbnail
        }
        new_diseases.append(new_disease)
    
    # 3. Save simplified diseases.json
    with open(diseases_file, 'w') as f:
        json.dump(new_diseases, f, indent=2)
    
    print(f"\n✓ Migration complete!")
    print(f"  Updated {len(new_diseases)} diseases")
    print(f"  Simplified diseases.json to only store: id, name, image")
    print(f"  Added 'name' field to case_history.json files")


if __name__ == '__main__':
    migrate()
