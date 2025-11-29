#!/usr/bin/env python3
"""
Diagnostic script to inspect LOCOMO and LongMemEval dataset structures.
Run this from your layer3_demos directory to see exact data formats.

Usage:
    python diagnose_datasets.py
"""

import json
from pathlib import Path


def diagnose_locomo(data_dir: Path):
    """Inspect LOCOMO dataset structure."""
    filepath = data_dir / "locomo.json"
    
    if not filepath.exists():
        print(f"[LOCOMO] File not found: {filepath}")
        print("         Run tests once to download, or download manually from:")
        print("         https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json")
        return
    
    print("=" * 70)
    print("LOCOMO DATASET ANALYSIS")
    print("=" * 70)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nDataset type: {type(data).__name__}")
    print(f"Number of items: {len(data) if isinstance(data, list) else 'N/A (dict)'}")
    
    if isinstance(data, list) and len(data) > 0:
        conv = data[0]
        print(f"\n--- First conversation ---")
        print(f"Top-level keys: {list(conv.keys())}")
        
        # Check conversation structure
        if "conversation" in conv:
            convo = conv["conversation"]
            print(f"\n'conversation' type: {type(convo).__name__}")
            
            if isinstance(convo, list):
                print(f"'conversation' length: {len(convo)}")
                if len(convo) > 0:
                    first_session = convo[0]
                    print(f"\nFirst session type: {type(first_session).__name__}")
                    if isinstance(first_session, dict):
                        print(f"First session keys: {list(first_session.keys())}")
                        
                        # Check for dialogue
                        if "dialogue" in first_session:
                            dialogue = first_session["dialogue"]
                            print(f"\n'dialogue' type: {type(dialogue).__name__}")
                            if isinstance(dialogue, list) and len(dialogue) > 0:
                                print(f"'dialogue' length: {len(dialogue)}")
                                print(f"First turn: {dialogue[0]}")
                        
                        # Check for session_id
                        if "session_id" in first_session:
                            print(f"\nsession_id: {first_session['session_id']}")
                    
                    # Print raw first session (truncated)
                    session_str = json.dumps(first_session, indent=2)
                    print(f"\n--- Raw first session (first 1500 chars) ---")
                    print(session_str[:1500])
                    
            elif isinstance(convo, dict):
                print(f"'conversation' keys: {list(convo.keys())[:20]}...")
                # Check for session_1, session_2, etc.
                session_keys = [k for k in convo.keys() if k.startswith("session_")]
                print(f"Session-like keys: {session_keys[:10]}")
        
        # Check QA structure
        if "qa" in conv:
            qa = conv["qa"]
            print(f"\n--- QA Structure ---")
            print(f"'qa' type: {type(qa).__name__}")
            if isinstance(qa, list):
                print(f"Number of questions: {len(qa)}")
                if len(qa) > 0:
                    print(f"First QA: {qa[0]}")
                    
                    # Count categories
                    categories = {}
                    for q in qa:
                        cat = q.get("category", "unknown")
                        categories[cat] = categories.get(cat, 0) + 1
                    print(f"Category distribution: {categories}")
        
        # Count total turns across all conversations
        print(f"\n--- Total Statistics ---")
        total_turns = 0
        total_questions = 0
        for i, c in enumerate(data):
            if "conversation" in c:
                convo = c["conversation"]
                if isinstance(convo, list):
                    for session in convo:
                        if isinstance(session, dict) and "dialogue" in session:
                            total_turns += len(session["dialogue"])
            if "qa" in c:
                total_questions += len(c["qa"])
        print(f"Total conversations: {len(data)}")
        print(f"Total turns: {total_turns}")
        print(f"Total questions: {total_questions}")


def diagnose_longmemeval(data_dir: Path):
    """Inspect LongMemEval dataset structure."""
    filepath = data_dir / "longmemeval_s.json"
    
    if not filepath.exists():
        print(f"\n[LongMemEval] File not found: {filepath}")
        print("              Run tests once to download, or download from HuggingFace:")
        print("              https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned")
        return
    
    print("\n" + "=" * 70)
    print("LONGMEMEVAL DATASET ANALYSIS")
    print("=" * 70)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nDataset type: {type(data).__name__}")
    print(f"Number of items: {len(data) if isinstance(data, list) else 'N/A'}")
    
    if isinstance(data, list) and len(data) > 0:
        item = data[0]
        print(f"\n--- First item ---")
        print(f"Top-level keys: {list(item.keys())}")
        
        # Key fields
        print(f"\nquestion_id: {item.get('question_id')}")
        print(f"question_type: {item.get('question_type')}")
        print(f"question: {item.get('question', '')[:100]}...")
        print(f"answer: {item.get('answer', '')[:100]}...")
        print(f"question_date: {item.get('question_date')}")
        
        # Haystack structure
        if "haystack_sessions" in item:
            sessions = item["haystack_sessions"]
            print(f"\nhaystack_sessions type: {type(sessions).__name__}")
            if isinstance(sessions, list):
                print(f"Number of sessions: {len(sessions)}")
                if len(sessions) > 0:
                    first_session = sessions[0]
                    print(f"First session type: {type(first_session).__name__}")
                    if isinstance(first_session, list) and len(first_session) > 0:
                        print(f"First session has {len(first_session)} turns")
                        print(f"First turn: {first_session[0]}")
        
        # Session IDs
        if "haystack_session_ids" in item:
            print(f"\nhaystack_session_ids: {item['haystack_session_ids'][:5]}...")
        
        if "answer_session_ids" in item:
            print(f"answer_session_ids: {item['answer_session_ids']}")
        
        # Question type distribution
        print(f"\n--- Question Type Distribution ---")
        q_types = {}
        abstention_count = 0
        for i in data:
            q_type = i.get("question_type", "unknown")
            q_types[q_type] = q_types.get(q_type, 0) + 1
            if i.get("question_id", "").endswith("_abs"):
                abstention_count += 1
        
        for q_type, count in sorted(q_types.items()):
            print(f"  {q_type}: {count}")
        print(f"  (abstention by ID suffix): {abstention_count}")


def main():
    # Determine data directory
    script_dir = Path(__file__).parent
    
    # Try common locations
    possible_dirs = [
        script_dir / "data",
        Path("data"),
        Path("/mnt/d/chillbot/prod_test/layer3_demos/data"),
    ]
    
    data_dir = None
    for d in possible_dirs:
        if d.exists():
            data_dir = d
            break
    
    if data_dir is None:
        print("Could not find data directory. Creating ./data/")
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
    
    print(f"Data directory: {data_dir.absolute()}")
    
    diagnose_locomo(data_dir)
    diagnose_longmemeval(data_dir)


if __name__ == "__main__":
    main()
