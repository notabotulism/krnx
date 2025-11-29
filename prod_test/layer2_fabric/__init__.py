"""
KRNX Layer 2 Fabric Tests - Corrected Version

All tests corrected to match actual KRNX API signatures:
- query_events() has NO 'order' parameter
- get_event(event_id) takes ONLY event_id
- remember() has NO 'timestamp' parameter
- SalienceEngine.compute() uses event_id, timestamp, etc. (not event object)
- TemporalEnricher.enrich() uses timestamp, previous_event, now (returns dict)

Test modules:
- test_f1_stream_ordering.py: F1 Stream Ordering Under Concurrency
- test_f2_cross_stream.py: F2 Cross-Stream Consistency
- test_f3_consumer_groups.py: F3 Consumer Group Delivery
- test_f4_replay.py: F4 Temporal Replay Equivalence
- test_f5_agent_membership.py: F5 Agent Join/Leave Handling
- test_f6_concurrent_safety.py: F6 Concurrent Read/Write Safety
- test_f7_backpressure.py: F7 Backpressure Behavior
- test_f8_f9_enrichment_episodes.py: F8 Enrichment & F9 Episode Boundaries
- test_f10_relation_scoring.py: F10 Relation Scoring Determinism
"""
