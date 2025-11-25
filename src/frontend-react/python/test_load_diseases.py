#!/usr/bin/env python3
"""
Test script to verify what load_diseases() returns
"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from api_manager import APIManager

# Test loading diseases
print("Testing load_diseases()...")
diseases = APIManager.load_diseases()

print(f"\nFound {len(diseases)} diseases")
print("\n" + "=" * 80)

for disease in diseases:
    print(f"\nDisease ID: {disease.get('id')}")
    print(f"Name: {disease.get('name')}")
    print(f"Description: {disease.get('description')[:100] if disease.get('description') else 'None'}...")
    print(f"Body Part: {disease.get('bodyPart')}")
    print(f"Map Position: {disease.get('mapPosition')}")
    print(f"Confidence: {disease.get('confidenceLevel')}")
    print(f"LLM Response: {disease.get('llmResponse')[:100] if disease.get('llmResponse') else 'None'}...")
    print(f"Timeline entries: {len(disease.get('timelineData', []))}")
    print(f"Has image: {bool(disease.get('image'))}")

print("\n" + "=" * 80)
print("\nFull JSON output:")
print(json.dumps(diseases, indent=2)[:2000] + "...")
