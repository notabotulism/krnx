# krnx Benchmark Results

**Date:** 2025-12-05T12:02:16.992048
**Python:** 3.10.12
**Platform:** Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35
**CPUs:** 4

## Summary

| Metric | Value |
|--------|-------|
| Peak Write Throughput | 5,994 events/sec |
| Read Latency (p50) | 0.878 ms |
| Read Latency (p99) | 1.425 ms |
| Disk Usage | 282.6 bytes/event |
| Correctness Tests | 6/6 passed |
| Security Tests | 18/18 passed |
| Hardening Tests | 11/11 passed |

## Performance Details

### Write Throughput

| Events | Throughput | Time |
|--------|------------|------|
| 100 | 5,994.38 events/sec | 0.017s |
| 1,000 | 4,177.57 events/sec | 0.239s |
| 10,000 | 3,917.07 events/sec | 2.553s |
| 50,000 | 3,855.60 events/sec | 12.968s |

### Read Latency

| Operation | p50 | p95 | p99 |
|-----------|-----|-----|-----|
| log() | 0.878 ms | 1.091 ms | 1.425 ms |
| show() | 0.022 ms | 0.041 ms | - ms |
| at() | 2.828 ms | 3.533 ms | - ms |

### Concurrent Writes

| Threads | Throughput | Integrity |
|---------|------------|-----------|
| 2 | 3,062.36 events/sec | PASS |
| 4 | 3,284.90 events/sec | PASS |
| 8 | 3,052.88 events/sec | PASS |
| 16 | 2,865.93 events/sec | PASS |

## Correctness Tests

| Test | Status | Details |
|------|--------|---------|
| hash_chain_integrity | ✓ PASS | 1000 events verified |
| branch_isolation | ✓ PASS | main: 2, isolated: 3 |
| concurrent_ordering | ✓ PASS | 400/400 events, chain valid |
| export_import_fidelity | ✓ PASS | 100 events |
| timestamp_ordering | ✓ PASS | at(ts+150): expected {1, 2, 4}, got {1, 2, 4} |
| verify_detects_corruption | ✓ PASS | tampered row 5 |

## Security Tests

### Data Leakage

| Test | Status | Details |
|------|--------|---------|
| workspace_isolation | ✓ PASS | no cross-workspace data access |
| branch_data_isolation | ✓ PASS | branches contain only their own data |
| deleted_branch_not_queryable | ✓ PASS | deleted branch hidden from normal queries |
| search_branch_isolation | ✓ PASS | search respects branch boundaries |

### Injection Attacks

| Test | Status | Details |
|------|--------|---------|
| sql_injection_content | ✓ PASS | 6 payloads stored safely |
| sql_injection_search | ✓ PASS | search handles injection attempts safely |
| sql_injection_branch_name | ✓ PASS | branch names sanitized |
| path_traversal_workspace | ✓ PASS | workspace confined to storage dir |

### Content Handling

| Test | Status | Details |
|------|--------|---------|
| malformed_json_resilience | ✓ PASS | 7 edge cases handled |
| unicode_content | ✓ PASS | 8 unicode samples preserved |
| binary_content | ✓ PASS | binary data preserved via base64 |
| large_content | ✓ PASS | 1MB content stored and retrieved |

### Integrity Attacks

| Test | Status | Details |
|------|--------|---------|
| hash_cannot_be_forged | ✓ PASS | content tampering detected |
| parent_chain_tampering | ✓ PASS | parent hash tampering detected |
| replay_attack_detection | ✓ PASS | event replay/duplication detected |

### Resource Exhaustion

| Test | Status | Details |
|------|--------|---------|
| large_event_count | ✓ PASS | 10000 events handled |
| many_branches | ✓ PASS | 100 branches handled |
| deep_content_nesting | ✓ PASS | 50 levels nested |

## Hardening Tests

### Crash Recovery

| Test | Status | Details |
|------|--------|---------|
| crash_recovery_mid_write | ✓ PASS | recovered 100 events, verify=True |
| crash_recovery_mid_batch | ✓ PASS | rolled back incomplete, 50 events intact |
| recovery_after_wal_checkpoint | ✓ PASS | 1100 events preserved |

### Soak Testing

| Test | Status | Details |
|------|--------|---------|
| soak_sustained_writes | ✓ PASS | 1,754,000 events in 600s, avg 2,937/sec (min 630, max 3,809), degradation ratio 0.69 |
| soak_memory_stability | ✓ PASS | memory growth: 0.1MB |

### Fuzzing

| Test | Status | Details |
|------|--------|---------|
| fuzz_event_types | ✓ PASS | 13 accepted, 0 rejected, db intact |
| fuzz_content_structure | ✓ PASS | 100/100 random structures handled |
| fuzz_agent_names | ✓ PASS | 10/10 agents handled |
| fuzz_branch_operations | ✓ PASS | 4 branches intact |
| fuzz_timestamps | ✓ PASS | 6/6 valid timestamps, db intact |
| fuzz_rapid_branch_switching | ✓ PASS | 506 events across 6 branches |
