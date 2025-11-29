#!/usr/bin/env python3
"""Quick check of session structure"""
import json
from pathlib import Path

data_dir = Path("/mnt/d/chillbot/prod_test/layer3_demos/data")
filepath = data_dir / "locomo.json"

with open(filepath, 'r') as f:
    data = json.load(f)

conv = data[0]
convo = conv["conversation"]

print("Speaker A:", convo.get("speaker_a"))
print("Speaker B:", convo.get("speaker_b"))
print()

# Get session_1
session_1 = convo.get("session_1", [])
print(f"session_1 type: {type(session_1).__name__}")
print(f"session_1 length: {len(session_1)}")

if session_1:
    print(f"\nFirst turn type: {type(session_1[0]).__name__}")
    print(f"First turn: {json.dumps(session_1[0], indent=2)}")
    print(f"\nSecond turn: {json.dumps(session_1[1], indent=2)}" if len(session_1) > 1 else "")

print(f"\nsession_1_date_time: {convo.get('session_1_date_time')}")

# Count all turns
total = 0
for key in convo:
    if key.startswith("session_") and not key.endswith("_date_time"):
        session = convo[key]
        if isinstance(session, list):
            total += len(session)
            
print(f"\nTotal turns in first conversation: {total}")
