#!/usr/bin/env python3
"""
Diagnostic script to check LOCOMO dataset structure.
Run this to see what keys exist in your locomo.json file.
"""

import json
from pathlib import Path

# Adjust this path if needed
DATA_PATH = Path("/mnt/d/chillbot/prod_test/layer3_demos/data/locomo.json")

def main():
    print(f"Loading: {DATA_PATH}")
    
    if not DATA_PATH.exists():
        print(f"ERROR: File not found: {DATA_PATH}")
        return
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nDataset type: {type(data)}")
    
    if isinstance(data, list):
        print(f"Number of conversations: {len(data)}")
        
        if len(data) > 0:
            print("\n" + "="*60)
            print("FIRST CONVERSATION STRUCTURE:")
            print("="*60)
            
            conv = data[0]
            print(f"\nTop-level keys: {list(conv.keys())}")
            
            # Look for session keys
            session_keys = [k for k in conv.keys() if k.startswith("session_")]
            print(f"\nSession keys found: {session_keys[:10]}...")  # First 10
            
            # Check first session
            for key in session_keys[:3]:
                if not key.endswith("_date_time") and not key.endswith("_summary"):
                    session_data = conv.get(key)
                    print(f"\n{key}:")
                    print(f"  Type: {type(session_data)}")
                    if isinstance(session_data, list):
                        print(f"  Length: {len(session_data)}")
                        if len(session_data) > 0:
                            print(f"  First item type: {type(session_data[0])}")
                            if isinstance(session_data[0], dict):
                                print(f"  First item keys: {list(session_data[0].keys())}")
                                print(f"  First item sample: {session_data[0]}")
                    elif isinstance(session_data, dict):
                        print(f"  Keys: {list(session_data.keys())}")
                    break
            
            # Check for alternative structures
            print("\n" + "="*60)
            print("CHECKING ALTERNATIVE STRUCTURES:")
            print("="*60)
            
            # Check for 'conversations' key
            if "conversations" in conv:
                print(f"\nFound 'conversations' key!")
                print(f"  Type: {type(conv['conversations'])}")
            
            # Check for 'dialogue' or 'dialog' keys
            for alt_key in ["dialogue", "dialog", "messages", "turns", "utterances"]:
                if alt_key in conv:
                    print(f"\nFound '{alt_key}' key!")
                    print(f"  Type: {type(conv[alt_key])}")
                    if isinstance(conv[alt_key], list) and len(conv[alt_key]) > 0:
                        print(f"  Length: {len(conv[alt_key])}")
                        print(f"  First item: {conv[alt_key][0]}")
            
            # Check for QA
            if "qa" in conv:
                print(f"\nFound 'qa' key with {len(conv['qa'])} questions")
                if len(conv['qa']) > 0:
                    print(f"  First QA: {conv['qa'][0]}")
            
            # Print raw first conversation (truncated)
            print("\n" + "="*60)
            print("RAW FIRST CONVERSATION (truncated):")
            print("="*60)
            conv_str = json.dumps(conv, indent=2)
            print(conv_str[:3000] + "..." if len(conv_str) > 3000 else conv_str)
            
    elif isinstance(data, dict):
        print(f"Top-level keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, list):
                print(f"    Length: {len(value)}")

if __name__ == "__main__":
    main()
