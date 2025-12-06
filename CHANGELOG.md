# Changelog

All notable changes to krnx will be documented in this file.

## [0.2.0] - 2025-12-06

### Added
- **Checkpoints** — Named save points for easy branching
  - `s.checkpoint(name, description)` — Create checkpoint
  - `s.checkpoints()` — List all checkpoints
  - `s.get_checkpoint(name)` — Get checkpoint details
  - `s.checkpoint_delete(name)` — Delete checkpoint
  - `s.branch_from_checkpoint(branch, checkpoint)` — Branch from checkpoint
  - CLI: `krnx checkpoint`, `krnx checkpoints`, `krnx checkpoint-delete`, `krnx branch-from-checkpoint`

- **Observability** — Built-in statistics
  - `s.stats()` — Event aggregation by type, agent, branch, time
  - Token counting from think events
  - CLI: `krnx stats`, `krnx stats --json`

- **Studio enhancements**
  - Focus-based navigation with Tab cycling
  - Larger timeline panel (18 lines)
  - Green focus borders for active panel
  - 6-character hash display with ◉ markers

- **Agent instrumentation**
  - Full LLM prompt/response recording in events
  - Complete audit trail for agent decisions

### Changed
- Test suite expanded to 24 tests (was 16)

## [0.1.0] - 2025-12-05

### Added
- Initial release
- Core substrate with SQLite + WAL mode
- Hash-chain integrity verification
- Branching and timeline management
- CLI with 13 commands
- Studio TUI for visual exploration
- Narrated demo mode
- Export/import JSONL
- Search and replay capabilities
