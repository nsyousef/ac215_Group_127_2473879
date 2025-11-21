#!/usr/bin/env python3
"""
Comprehensive test showing what the UI will receive
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from api_manager import APIManager

print("="*80)
print("TESTING FULL DATA FLOW")
print("="*80)

# Test 1: Load diseases (what UI gets initially)
print("\n1. LOADING DISEASES (Initial UI State)")
print("-"*80)
diseases = APIManager.load_diseases()
print(f"Found {len(diseases)} disease(s)")

for disease in diseases:
    print(f"\nDisease: {disease['name']}")
    print(f"  ID: {disease['id']}")
    print(f"  Body Part: {disease['bodyPart']}")
    print(f"  Map Position: {disease['mapPosition']}")
    print(f"  Confidence: {disease['confidenceLevel']}%")
    print(f"  Description: {disease['description'][:80]}..." if disease['description'] else "  Description: None")
    print(f"  LLM Response: {disease['llmResponse'][:80]}..." if disease['llmResponse'] else "  LLM Response: None")
    print(f"  Timeline Entries: {len(disease['timelineData'])}")

# Test 2: Load case history (what TimeTrackingPanel gets)
print("\n\n2. LOADING CASE HISTORY (TimeTrackingPanel)")
print("-"*80)
if diseases:
    case_id = diseases[0]['id']
    case_history = APIManager.load_case_history(f"case_{case_id}")
    
    dates = case_history.get('dates', {})
    print(f"Found {len(dates)} timeline entry/entries")
    
    for date, entry in dates.items():
        print(f"\n  Date: {date}")
        print(f"    Image Path: {entry.get('image_path', 'None')}")
        print(f"    Text Summary: {entry.get('text_summary', 'None')[:80]}...")
        print(f"    Predictions: {list(entry.get('predictions', {}).keys())}")

# Test 3: Load conversation history (what ChatPanel gets)
print("\n\n3. LOADING CONVERSATION HISTORY (ChatPanel)")
print("-"*80)
if diseases:
    case_id = diseases[0]['id']
    conversation_file = Path(__file__).parent / f"case_{case_id}" / "conversation_history.json"
    
    if conversation_file.exists():
        with open(conversation_file, 'r') as f:
            conversation = json.load(f)
        
        print(f"Found {len(conversation)} conversation entry/entries")
        
        for i, entry in enumerate(conversation):
            print(f"\n  Entry {i+1}:")
            if entry.get('user'):
                print(f"    User: {entry['user']['message'][:80]}...")
            if entry.get('llm'):
                print(f"    LLM: {entry['llm']['message'][:80]}...")
    else:
        print("No conversation history file found")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
