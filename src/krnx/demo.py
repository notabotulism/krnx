"""
krnx demo — Git for ML Agent State

A 90-second self-running demo showing:
1. Agent runs, records decisions
2. Bad outcome ($8k loss)
3. Query timeline, find root cause
4. Branch from before the mistake
5. Replay with correct data
6. Compare outcomes
7. Verify both timelines

Usage:
    krnx demo
    
Controls:
    SPACE   Pause/Resume
    R       Restart
    Q       Quit
"""

import json
import time
import tempfile
import shutil
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Static, RichLog
from textual.containers import Horizontal, Vertical
from textual.binding import Binding

from . import init as krnx_init, Substrate


# =============================================================================
# DEMO SCRIPT
# =============================================================================

@dataclass
class Step:
    """A demo step."""
    delay: float          # Seconds before this step
    phase: str            # Phase name
    action: str           # What to do
    narration: str        # What to show
    code: str = ""        # Code to display
    event_type: str = ""
    content: dict = None
    agent: str = "AGENT"
    branch: str = "main"
    highlight: str = ""   # "error", "success", "info"


SCRIPT: List[Step] = [
    # === INTRO ===
    Step(1.0, "INTRO", "narrate",
         "krnx — git for ml agent state",
         highlight="info"),
    Step(2.0, "INTRO", "narrate",
         "A refund agent processes a request..."),
    
    # === AGENT RUNS ===
    Step(1.5, "RECORD", "narrate",
         "─── AGENT RUNS ───"),
    
    Step(1.2, "RECORD", "event",
         "👁 observe: Customer requests $8,241 refund",
         code='s.record("observe", {"request": "refund", "amount": 8241})',
         event_type="observe",
         content={"request": "refund", "amount": 8241, "customer": "4521"},
         agent="INTAKE"),
    
    Step(1.2, "RECORD", "event",
         "🔍 observe: Lookup says PREMIUM tier, $10k limit",
         code='s.record("observe", {"tier": "PREMIUM", "limit": 10000})',
         event_type="observe",
         content={"tier": "PREMIUM", "limit": 10000, "source": "cache"},
         agent="LOOKUP",
         highlight="error"),  # This is the bug - stale cache
    
    Step(1.2, "RECORD", "event",
         "💭 think: Premium user, high limit, approve",
         code='s.record("think", {"decision": "approve", "confidence": 0.94})',
         event_type="think",
         content={"reasoning": "Premium user with $10k limit, request is within bounds", "confidence": 0.94},
         agent="PLANNER"),
    
    Step(1.2, "RECORD", "event",
         "⚡ act: APPROVE $8,241 refund",
         code='s.record("act", {"action": "approve", "amount": 8241})',
         event_type="act",
         content={"action": "approve", "amount": 8241},
         agent="EXECUTOR"),
    
    Step(1.5, "RECORD", "event",
         "❌ result: LOSS $8,141 — actual tier was BASIC ($100 limit)",
         code='s.record("result", {"outcome": "LOSS", "amount": 8141})',
         event_type="result",
         content={"outcome": "LOSS", "loss": 8141, "actual_tier": "BASIC", "actual_limit": 100},
         agent="SYSTEM",
         highlight="error"),
    
    Step(2.0, "RECORD", "narrate",
         "The agent made a costly mistake. What happened?",
         highlight="error"),
    
    # === INVESTIGATE ===
    Step(2.0, "QUERY", "narrate",
         "─── INVESTIGATE ───"),
    
    Step(1.5, "QUERY", "query",
         "📋 krnx log shows 5 events on main branch",
         code="krnx log"),
    
    Step(1.5, "QUERY", "query",
         "🔍 krnx search finds the approval decision",
         code='krnx search "approve"'),
    
    Step(1.5, "QUERY", "query",
         "⏪ krnx at shows state before the decision",
         code="krnx at evt_xxx"),
    
    Step(2.0, "QUERY", "narrate",
         "Root cause: LOOKUP returned stale cache (PREMIUM instead of BASIC)",
         highlight="info"),
    
    # === BRANCH ===
    Step(2.0, "BRANCH", "narrate",
         "─── BRANCH & REPLAY ───"),
    
    Step(1.5, "BRANCH", "branch",
         "⎇ Create branch 'fix' from before the bad lookup",
         code='krnx branch fix --from evt_intake'),
    
    Step(1.5, "BRANCH", "narrate",
         "Now replay with correct data...",
         highlight="info"),
    
    # === REPLAY ON BRANCH ===
    Step(1.2, "REPLAY", "event",
         "🔍 observe: [fix] Correct lookup: BASIC tier, $100 limit",
         code='s.record("observe", {"tier": "BASIC"}, branch="fix")',
         event_type="observe",
         content={"tier": "BASIC", "limit": 100, "source": "fresh"},
         agent="LOOKUP",
         branch="fix",
         highlight="success"),
    
    Step(1.2, "REPLAY", "event",
         "💭 think: [fix] Basic user, limit exceeded, must deny",
         code='s.record("think", {"decision": "deny"}, branch="fix")',
         event_type="think",
         content={"reasoning": "Basic user with $100 limit, request exceeds bounds", "decision": "deny"},
         agent="PLANNER",
         branch="fix"),
    
    Step(1.2, "REPLAY", "event",
         "⚡ act: [fix] DENY refund request",
         code='s.record("act", {"action": "deny"}, branch="fix")',
         event_type="act",
         content={"action": "deny", "reason": "exceeds_limit"},
         agent="EXECUTOR",
         branch="fix"),
    
    Step(1.2, "REPLAY", "event",
         "✓ result: [fix] OK — $0 loss, policy enforced",
         code='s.record("result", {"outcome": "OK"}, branch="fix")',
         event_type="result",
         content={"outcome": "OK", "loss": 0},
         agent="SYSTEM",
         branch="fix",
         highlight="success"),
    
    # === VERIFY ===
    Step(2.0, "VERIFY", "narrate",
         "─── VERIFY & COMPARE ───"),
    
    Step(1.5, "VERIFY", "verify",
         "⛓ main: ✓ hash chain intact (5 events)",
         code="krnx verify --branch main"),
    
    Step(1.2, "VERIFY", "verify",
         "⛓ fix:  ✓ hash chain intact (5 events)",
         code="krnx verify --branch fix"),
    
    # === DIFF ===
    Step(1.5, "COMPARE", "diff",
         "📊 krnx diff main..fix",
         code="krnx diff main..fix"),
    
    Step(1.5, "COMPARE", "narrate",
         "main: PREMIUM → approve → LOSS $8,141",
         highlight="error"),
    
    Step(1.2, "COMPARE", "narrate",
         "fix:  BASIC → deny → OK $0",
         highlight="success"),
    
    # === CONCLUSION ===
    Step(2.5, "DONE", "narrate",
         "Same agent. Same code. Different context. Different outcome.",
         highlight="info"),
    
    Step(2.0, "DONE", "narrate",
         "────────────────────────────────────────"),
    
    Step(1.0, "DONE", "narrate",
         "pip install krnx",
         highlight="info"),
    
    Step(1.5, "DONE", "narrate",
         "Put your agents in production with confidence."),
]


# =============================================================================
# STYLES
# =============================================================================

CSS = """
Screen {
    background: #0a0f0c;
}

#phase-bar {
    dock: top;
    height: 1;
    background: #1a2420;
    color: #6bbd8a;
    padding: 0 2;
}

#main-container {
    height: 1fr;
}

#left-panel {
    width: 1fr;
    border-right: solid #2a3a30;
    padding: 1 2;
}

#right-panel {
    width: 55;
    padding: 1 2;
    background: #0d1410;
}

#event-log {
    height: 100%;
}

#code-panel {
    height: 100%;
}

#graph-bar {
    height: 5;
    border-bottom: solid #2a3a30;
    padding: 1 2;
    background: #0d1410;
}

#bottom-bar {
    dock: bottom;
    height: 3;
    background: #141c17;
    border-top: solid #2a3a30;
    padding: 0 2;
}

#hash-display {
    height: 1;
    color: #4a6a5a;
}

#status-bar {
    height: 1;
    color: #7a9a8a;
}

.error {
    color: #d47c7c;
}

.success {
    color: #7cd47c;
}

.info {
    color: #7cb7d4;
}

.dim {
    color: #4a5a50;
}

.branch-main {
    color: #6bbd8a;
}

.branch-fix {
    color: #c4956a;
}
"""


# =============================================================================
# DEMO APP
# =============================================================================

class DemoApp(App):
    """krnx demo application."""
    
    CSS = CSS
    TITLE = "krnx demo"
    
    BINDINGS = [
        Binding("space", "toggle_pause", "Pause/Resume"),
        Binding("r", "restart", "Restart"),
        Binding("q", "quit", "Quit"),
    ]
    
    def __init__(self):
        super().__init__()
        self.step_index = 0
        self.elapsed = 0.0
        self.paused = False
        self.next_step_at = 0.0
        
        # Temp directory for demo data
        self.temp_dir = tempfile.mkdtemp(prefix="krnx_demo_")
        self.substrate: Optional[Substrate] = None
        
        self.events_main: List = []
        self.events_fix: List = []
        self.current_phase = "INTRO"
    
    def compose(self) -> ComposeResult:
        yield Static("", id="phase-bar")
        with Vertical(id="graph-bar"):
            yield RichLog(id="graph-display", markup=True)
        with Horizontal(id="main-container"):
            with Vertical(id="left-panel"):
                yield RichLog(id="event-log", markup=True)
            with Vertical(id="right-panel"):
                yield RichLog(id="code-panel", markup=True)
        with Vertical(id="bottom-bar"):
            yield Static("", id="hash-display")
            yield Static("", id="status-bar")
    
    def on_mount(self):
        self._event_log = self.query_one("#event-log", RichLog)
        self._code_panel = self.query_one("#code-panel", RichLog)
        self._phase_bar = self.query_one("#phase-bar", Static)
        self._hash_display = self.query_one("#hash-display", Static)
        self._status_bar = self.query_one("#status-bar", Static)
        self._graph_display = self.query_one("#graph-display", RichLog)
        
        # Initialize substrate
        self.substrate = krnx_init("demo", path=self.temp_dir)
        
        # Initial display
        self._event_log.write("[dim]krnx demo[/dim]\n")
        self._code_panel.write("[dim]# commands[/dim]\n")
        
        self.update_graph()
        self.update_display()
        self.set_interval(0.1, self.tick)
    
    def tick(self):
        """Main loop tick."""
        if self.paused:
            return
        
        self.elapsed += 0.1
        
        if self.step_index >= len(SCRIPT):
            self.current_phase = "DONE"
            self.update_display()
            return
        
        step = SCRIPT[self.step_index]
        
        if self.elapsed >= self.next_step_at:
            self.execute_step(step)
            self.step_index += 1
            
            if self.step_index < len(SCRIPT):
                self.next_step_at = self.elapsed + SCRIPT[self.step_index].delay
        
        self.update_display()
    
    def execute_step(self, step: Step):
        """Execute a demo step."""
        self.current_phase = step.phase
        
        # Show code
        if step.code:
            self._code_panel.write(f"[#6bbd8a]{step.code}[/]")
        
        # Style based on highlight
        style = ""
        if step.highlight == "error":
            style = "#d47c7c"
        elif step.highlight == "success":
            style = "#7cd47c"
        elif step.highlight == "info":
            style = "#7cb7d4"
        else:
            style = "#9aaa9e"
        
        # Execute action
        if step.action == "event":
            event_id = self.substrate.record(
                step.event_type,
                step.content,
                agent=step.agent,
                branch=step.branch,
            )
            event = self.substrate.show(event_id)
            
            if step.branch == "main":
                self.events_main.append(event)
            else:
                self.events_fix.append(event)
            
            hash_short = event.hash[:8]
            branch_marker = f"[#c4956a][fix][/] " if step.branch != "main" else ""
            
            self._event_log.write(
                f"{branch_marker}[{style}]{step.narration}[/] [#4a6a5a][{hash_short}][/]"
            )
            
            self.update_graph()
            self.update_hash_display()
        
        elif step.action == "branch":
            if self.events_main:
                try:
                    self.substrate.branch("fix", from_event=self.events_main[0].id)
                except:
                    pass  # Branch may exist
            self._event_log.write(f"\n[{style}]{step.narration}[/]")
            self.update_graph()
        
        elif step.action == "narrate":
            if "───" in step.narration:
                self._event_log.write(f"\n[#e8ede9]{step.narration}[/]")
            else:
                self._event_log.write(f"[{style}]{step.narration}[/]")
        
        elif step.action in ("query", "verify", "diff"):
            self._event_log.write(f"[{style}]{step.narration}[/]")
    
    def update_graph(self):
        """Update the branch graph visualization."""
        self._graph_display.clear()
        
        # Build graph
        main_nodes = "".join(["●─" for _ in self.events_main]) or "○"
        fix_nodes = "".join(["●─" for _ in self.events_fix])
        
        main_line = f"[#6bbd8a]main[/]  ─{main_nodes}→"
        self._graph_display.write(main_line)
        
        if self.events_fix:
            # Fork point visualization
            fork_pos = min(1, len(self.events_main) - 1) if self.events_main else 0
            indent = "        " + "──" * fork_pos
            fix_line = f"{indent}└─[#c4956a]{fix_nodes}→ fix[/]"
            self._graph_display.write(fix_line)
    
    def update_hash_display(self):
        """Update hash chain display."""
        parts = []
        
        if self.events_main:
            recent = self.events_main[-4:]
            chain = " → ".join(e.hash[:6] for e in recent)
            parts.append(f"[#6bbd8a]main:[/] {chain}")
        
        if self.events_fix:
            recent = self.events_fix[-3:]
            chain = " → ".join(e.hash[:6] for e in recent)
            parts.append(f"[#c4956a]fix:[/] {chain}")
        
        self._hash_display.update("  ".join(parts))
    
    def update_display(self):
        """Update status displays."""
        phase_labels = {
            "INTRO": "krnx demo",
            "RECORD": "▶ RECORD — Agent makes decisions",
            "QUERY": "🔍 QUERY — Investigate what happened",
            "BRANCH": "⎇ BRANCH — Fork the timeline",
            "REPLAY": "↺ REPLAY — Try different context",
            "VERIFY": "⛓ VERIFY — Prove integrity",
            "COMPARE": "📊 COMPARE — See the difference",
            "DONE": "✓ DONE — SPACE restart, Q quit"
        }
        self._phase_bar.update(phase_labels.get(self.current_phase, ""))
        
        pause_str = "❚❚ PAUSED" if self.paused else "▶"
        self._status_bar.update(
            f"{pause_str}  {self.elapsed:.1f}s  │  SPACE pause  R restart  Q quit"
        )
    
    def action_toggle_pause(self):
        self.paused = not self.paused
        self.update_display()
    
    def action_restart(self):
        # Cleanup and reset
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir = tempfile.mkdtemp(prefix="krnx_demo_")
        self.substrate = krnx_init("demo", path=self.temp_dir)
        
        self.step_index = 0
        self.elapsed = 0.0
        self.next_step_at = 0.0
        self.paused = False
        self.events_main = []
        self.events_fix = []
        self.current_phase = "INTRO"
        
        self._event_log.clear()
        self._code_panel.clear()
        self._graph_display.clear()
        self._event_log.write("[dim]krnx demo[/dim]\n")
        self._code_panel.write("[dim]# commands[/dim]\n")
        self._hash_display.update("")
        self.update_graph()
        self.update_display()
    
    def action_quit(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.exit()


def run_demo():
    """Run the krnx demo."""
    app = DemoApp()
    app.run()


if __name__ == "__main__":
    run_demo()
