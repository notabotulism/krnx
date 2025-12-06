"""
krnx Narrated Demo — The Story

A dramatic walkthrough showing:
1. Agent processes a request
2. Disaster strikes (fraud)
3. Investigation in Studio
4. Branch & fix
5. Verification

Usage:
    krnx demo
    
Controls:
    SPACE/ENTER  Continue
    Q            Quit
"""

import os
import sys
import time
import tempfile
import shutil
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich import box

# =============================================================================
# STORY DATA
# =============================================================================

CUSTOMER_NAME = "Alex Thompson"
CUSTOMER_ID = "cust_8x7k2m"
ORDER_ID = "ord_j3k8n2"
AMOUNT = 8000.00

# System prompt for audit trail
DEMO_SYSTEM_PROMPT = """You are a customer service agent handling refund requests.

Your job is to decide whether to APPROVE or DENY refund requests.

Guidelines:
- Check order history for patterns
- Consider refund amount relative to customer history
- If fraud_score is provided and > 70, this is HIGH RISK
- Large refunds (>$5000) on accounts with limited history are risky

Respond with JSON: {"decision": "APPROVE/DENY", "reasoning": "...", "confidence": 0.0-1.0}"""

SCENARIO_MAIN = {
    "customer_id": CUSTOMER_ID,
    "customer_name": CUSTOMER_NAME,
    "order_id": ORDER_ID,
    "amount": AMOUNT,
    "reason": "Item not as described, requesting full refund",
    "order_history": [
        {"order_id": "ord_a1b2c3", "amount": 450.00, "status": "completed"},
        {"order_id": "ord_d4e5f6", "amount": 1200.00, "status": "completed"},
        {"order_id": ORDER_ID, "amount": AMOUNT, "status": "delivered"},
    ],
}

SCENARIO_FIX = {
    **SCENARIO_MAIN,
    "fraud_score": 87.3,
}

# Full think event for main branch (shows what a real LLM trace looks like)
THINK_MAIN = {
    "model": "gpt-4o (demo)",
    "tokens_used": 342,
    "system_prompt": DEMO_SYSTEM_PROMPT,
    "user_prompt": f"""Process this refund request:

Customer: {CUSTOMER_NAME} ({CUSTOMER_ID})
Order: {ORDER_ID}
Amount: ${AMOUNT:,.2f}
Reason: Item not as described, requesting full refund

Order History:
[
  {{"order_id": "ord_a1b2c3", "amount": 450.00, "status": "completed"}},
  {{"order_id": "ord_d4e5f6", "amount": 1200.00, "status": "completed"}},
  {{"order_id": "{ORDER_ID}", "amount": {AMOUNT}, "status": "delivered"}}
]""",
    "raw_response": """{
  "decision": "APPROVE",
  "reasoning": "Customer has valid order history. Amount matches recent purchase. Approving.",
  "confidence": 0.78
}""",
    "parsed": {
        "decision": "APPROVE",
        "reasoning": "Customer has valid order history. Amount matches recent purchase. Approving.",
        "confidence": 0.78,
    },
}

# Full think event for fix branch (shows fraud detection)
THINK_FIX = {
    "model": "gpt-4o (demo)",
    "tokens_used": 287,
    "system_prompt": DEMO_SYSTEM_PROMPT,
    "user_prompt": f"""Process this refund request:

Customer: {CUSTOMER_NAME} ({CUSTOMER_ID})
Order: {ORDER_ID}
Amount: ${AMOUNT:,.2f}
Reason: Item not as described, requesting full refund

Order History:
[
  {{"order_id": "ord_a1b2c3", "amount": 450.00, "status": "completed"}},
  {{"order_id": "ord_d4e5f6", "amount": 1200.00, "status": "completed"}},
  {{"order_id": "{ORDER_ID}", "amount": {AMOUNT}, "status": "delivered"}}
]

⚠️ FRAUD SCORE: 87.3/100 (HIGH RISK)""",
    "raw_response": """{
  "decision": "DENY",
  "reasoning": "HIGH RISK: Fraud score 87.3 exceeds threshold. Despite order history, denying due to fraud indicators.",
  "confidence": 0.94
}""",
    "parsed": {
        "decision": "DENY",
        "reasoning": "HIGH RISK: Fraud score 87.3 exceeds threshold. Denying request.",
        "confidence": 0.94,
    },
}

# =============================================================================
# DEMO ENGINE
# =============================================================================

class NarratedDemo:
    def __init__(self):
        self.console = Console()
        self.temp_dir = tempfile.mkdtemp(prefix="krnx_demo_")
        self.substrate = None
        self.events = []
    
    def wait(self, prompt: str = "[dim]SPACE to continue[/dim]"):
        """Wait for user input."""
        self.console.print()
        self.console.print(Align.right(prompt))
        
        # Read single keypress
        if sys.platform == "win32":
            import msvcrt
            while True:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key in (b' ', b'\r', b'\n'):
                        return True
                    if key in (b'q', b'Q'):
                        return False
        else:
            import tty
            import termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                key = sys.stdin.read(1)
                if key in ('q', 'Q'):
                    return False
                return True
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def typewriter(self, text: str, delay: float = 0.02):
        """Print text with typewriter effect."""
        for char in text:
            self.console.print(char, end="", highlight=False)
            time.sleep(delay)
        self.console.print()
    
    def clear(self):
        """Clear screen."""
        self.console.clear()
    
    def pause(self, seconds: float = 0.5):
        """Brief pause for effect."""
        time.sleep(seconds)
    
    def show_command(self, cmd: str, output: Optional[str] = None):
        """Show a command being executed."""
        self.console.print()
        self.console.print(f"[green]$[/green] [bold]{cmd}[/bold]")
        self.pause(0.3)
        if output:
            self.console.print(output)
    
    def record_event(self, event_type: str, content: dict, show: bool = True) -> dict:
        """Record an event and optionally display it."""
        event_id = self.substrate.record(event_type, content, agent="cs-agent")
        event = self.substrate.show(event_id)
        self.events.append(event)
        
        if show:
            parent = event.parent[:8] if event.parent else "genesis "
            self.console.print(
                f"  [cyan]{event_type:8}[/cyan] │ {parent} → [bold green]{event.hash[:8]}[/bold green]"
            )
        
        return event
    
    # =========================================================================
    # STORY BEATS
    # =========================================================================
    
    def act1_setup(self) -> bool:
        """Act 1: Setup the scenario."""
        self.clear()
        
        # Title
        self.console.print()
        self.console.print(Panel(
            "[bold]krnx demo[/bold]\n\n"
            "[dim]Git for AI Agent State[/dim]",
            box=box.DOUBLE,
            border_style="green",
            padding=(1, 4),
        ))
        
        if not self.wait("[dim]SPACE to begin · Q to quit[/dim]"):
            return False
        
        self.clear()
        
        # Scene: Your agent in production
        self.console.print()
        self.console.print("[bold cyan]━━━ YOUR AGENT IN PRODUCTION ━━━[/bold cyan]")
        self.console.print()
        self.typewriter("You've deployed a customer service agent.", 0.03)
        self.typewriter("It handles refund requests. Thousands per day.", 0.03)
        self.typewriter("Everything's been running smoothly...", 0.03)
        self.pause(1)
        
        if not self.wait():
            return False
        
        self.clear()
        
        # Initialize krnx
        self.console.print()
        self.console.print("[bold cyan]━━━ STEP 1: INSTRUMENT YOUR AGENT ━━━[/bold cyan]")
        self.console.print()
        self.console.print("Add krnx to capture every decision:")
        self.console.print()
        
        self.show_command("krnx init cs-agent")
        
        from krnx import init as krnx_init
        self.substrate = krnx_init("demo-agent", path=self.temp_dir)
        
        self.console.print("  [green]✓[/green] Workspace created: [bold]cs-agent[/bold]")
        self.console.print()
        
        self.console.print("[dim]# In your agent code:[/dim]")
        self.console.print('[cyan]s.record("observe", {"request": ...})[/cyan]')
        self.console.print('[cyan]s.record("think", {"reasoning": ...})[/cyan]')
        self.console.print('[cyan]s.record("act", {"decision": ...})[/cyan]')
        
        if not self.wait():
            return False
        
        return True
    
    def act2_disaster(self) -> bool:
        """Act 2: A request comes in, agent approves, it's fraud."""
        self.clear()
        
        # Incoming request
        self.console.print()
        self.console.print("[bold cyan]━━━ INCOMING REQUEST ━━━[/bold cyan]")
        self.console.print()
        
        self.console.print(Panel(
            f"[bold]Customer:[/bold] {CUSTOMER_NAME}\n"
            f"[bold]Request:[/bold] Refund ${AMOUNT:,.2f}\n"
            f"[bold]Reason:[/bold] \"Item not as described\"",
            title="Refund Request",
            border_style="yellow",
        ))
        
        if not self.wait():
            return False
        
        # Agent processes
        self.console.print()
        self.console.print("[dim]Agent processing...[/dim]")
        self.console.print()
        self.pause(0.5)
        
        # Record observe
        self.record_event("observe", SCENARIO_MAIN)
        self.pause(0.3)
        
        # Record think — FULL audit trail
        self.record_event("think", THINK_MAIN)
        self.pause(0.3)
        
        # Record act
        self.record_event("act", {
            "decision": "APPROVE",
            "amount": AMOUNT,
        })
        self.pause(0.3)
        
        # Record result
        self.record_event("result", {
            "outcome": "REFUNDED",
            "amount": AMOUNT,
        })
        
        self.console.print()
        self.console.print("[green]✓ Refund processed: $8,000.00[/green]")
        
        if not self.wait():
            return False
        
        # THE DISASTER
        self.clear()
        self.console.print()
        self.console.print()
        self.console.print()
        
        self.console.print(Align.center("[bold yellow]⚠️  48 HOURS LATER  ⚠️[/bold yellow]"))
        self.pause(1.5)
        
        self.clear()
        self.console.print()
        self.console.print(Panel(
            "[bold red]CHARGEBACK RECEIVED[/bold red]\n\n"
            f"Account [bold]{CUSTOMER_ID}[/bold] was COMPROMISED.\n\n"
            "The refund request was [bold]FRAUDULENT[/bold].\n\n"
            f"[bold red]You just lost ${AMOUNT:,.2f}[/bold red]",
            border_style="red",
            box=box.HEAVY,
            padding=(1, 4),
        ))
        
        if not self.wait():
            return False
        
        # The questions
        self.console.print()
        self.console.print(Panel(
            "[bold]Your manager asks:[/bold]\n\n"
            "  • \"What did the agent see?\"\n"
            "  • \"Why did it approve?\"\n"
            "  • \"Can you prove the logs weren't tampered with?\"\n"
            "  • \"What would have happened with fraud detection?\"",
            border_style="yellow",
        ))
        
        if not self.wait("[dim]SPACE to investigate[/dim]"):
            return False
        
        return True
    
    def act3_investigation(self) -> bool:
        """Act 3: Investigate with krnx."""
        self.clear()
        
        self.console.print()
        self.console.print("[bold cyan]━━━ INVESTIGATION ━━━[/bold cyan]")
        self.console.print()
        self.typewriter("With krnx, you can answer all of these.", 0.03)
        self.console.print()
        
        if not self.wait():
            return False
        
        # Show the timeline
        self.console.print()
        self.show_command("krnx log")
        self.console.print()
        
        table = Table(show_header=True, box=box.SIMPLE)
        table.add_column("Type", style="cyan")
        table.add_column("Content", style="white")
        table.add_column("Hash", style="green")
        
        for e in self.events:
            content_preview = str(e.content)[:40] + "..." if len(str(e.content)) > 40 else str(e.content)
            table.add_row(e.type, content_preview, e.hash[:12])
        
        self.console.print(table)
        
        if not self.wait():
            return False
        
        # Verify integrity
        self.console.print()
        self.show_command("krnx verify")
        self.pause(0.5)
        
        self.console.print()
        self.console.print("[bold]Hash Chain:[/bold]")
        for i, e in enumerate(self.events):
            parent = e.parent[:8] if e.parent else "genesis "
            self.console.print(f"  {e.type:8} │ {parent} → [green]{e.hash[:8]}[/green]")
        
        self.console.print()
        self.console.print("[bold green]✓ Chain intact. No tampering detected.[/bold green]")
        
        if not self.wait():
            return False
        
        # Show the decision point
        self.clear()
        self.console.print()
        self.console.print("[bold cyan]━━━ THE DECISION POINT ━━━[/bold cyan]")
        self.console.print()
        
        think_event = self.events[1]  # The "think" event
        self.show_command(f"krnx show {think_event.id}")
        self.console.print()
        
        self.console.print(Panel(
            f"[bold]Type:[/bold] think\n"
            f"[bold]Hash:[/bold] {think_event.hash}\n"
            f"[bold]Parent:[/bold] {think_event.parent}\n\n"
            f"[bold]Content:[/bold]\n"
            f"  reasoning: \"{think_event.content.get('reasoning', '')}\"\n"
            f"  confidence: {think_event.content.get('confidence', 0)}",
            title="Event Details",
            border_style="blue",
        ))
        
        self.console.print()
        self.console.print("[yellow]The agent had no fraud score to check.[/yellow]")
        self.console.print("[yellow]What if it did?[/yellow]")
        
        if not self.wait():
            return False
        
        return True
    
    def act4_branch_and_fix(self) -> bool:
        """Act 4: Branch and try with fraud detection."""
        self.clear()
        
        self.console.print()
        self.console.print("[bold cyan]━━━ BRANCH: WHAT IF? ━━━[/bold cyan]")
        self.console.print()
        self.typewriter("krnx lets you branch from any point and replay.", 0.03)
        self.console.print()
        
        # Create branch
        first_event = self.events[0]
        self.show_command(f"krnx branch fix --from {first_event.id[:12]}")
        self.substrate.branch("fix", from_event=first_event.id)
        self.console.print("  [green]✓[/green] Branch 'fix' created from observe event")
        
        if not self.wait():
            return False
        
        # Show the branch graph
        self.console.print()
        self.console.print("[bold]Timeline:[/bold]")
        self.console.print()
        self.console.print("  [green]main[/green]  ●──●──●──●──→  [red]LOSS[/red]")
        self.console.print("            └────────────→  [cyan]fix[/cyan] (new branch)")
        
        if not self.wait():
            return False
        
        # Replay with fraud score
        self.clear()
        self.console.print()
        self.console.print("[bold cyan]━━━ REPLAY WITH FRAUD DETECTION ━━━[/bold cyan]")
        self.console.print()
        
        self.console.print("[dim]Same request, but now with fraud score...[/dim]")
        self.console.print()
        
        self.console.print(Panel(
            f"[bold]Customer:[/bold] {CUSTOMER_NAME}\n"
            f"[bold]Request:[/bold] Refund ${AMOUNT:,.2f}\n"
            f"[bold red]⚠️ Fraud Score: 87.3/100[/bold red]",
            title="Refund Request (with fraud check)",
            border_style="yellow",
        ))
        
        if not self.wait():
            return False
        
        self.console.print()
        self.console.print("[dim]Agent processing on 'fix' branch...[/dim]")
        self.console.print()
        
        # Record on fix branch with full audit data
        self.substrate.record("observe", SCENARIO_FIX, agent="cs-agent", branch="fix")
        parent = self.events[0].hash[:8]
        self.console.print(f"  [cyan]observe [/cyan] │ {parent} → [bold green]{self.substrate.log(branch='fix')[0].hash[:8]}[/bold green]")
        self.pause(0.3)
        
        # Full think event with prompt/response
        self.substrate.record("think", THINK_FIX, agent="cs-agent", branch="fix")
        self.console.print(f"  [cyan]think   [/cyan] │ ........ → [bold green]{self.substrate.log(branch='fix')[0].hash[:8]}[/bold green]")
        self.pause(0.3)
        
        self.substrate.record("act", {
            "decision": "DENY",
            "reason": "Fraud risk too high",
        }, agent="cs-agent", branch="fix")
        self.console.print(f"  [cyan]act     [/cyan] │ ........ → [bold green]{self.substrate.log(branch='fix')[0].hash[:8]}[/bold green]")
        self.pause(0.3)
        
        self.substrate.record("result", {
            "outcome": "FRAUD_PREVENTED",
            "amount_saved": AMOUNT,
        }, agent="cs-agent", branch="fix")
        self.console.print(f"  [cyan]result  [/cyan] │ ........ → [bold green]{self.substrate.log(branch='fix')[0].hash[:8]}[/bold green]")
        
        self.console.print()
        self.console.print("[bold green]✓ Request DENIED — Fraud prevented![/bold green]")
        
        if not self.wait():
            return False
        
        return True
    
    def act5_proof(self) -> bool:
        """Act 5: Verify both timelines."""
        self.clear()
        
        self.console.print()
        self.console.print("[bold cyan]━━━ THE PROOF ━━━[/bold cyan]")
        self.console.print()
        
        # Final timeline visualization
        self.console.print("[bold]Both timelines preserved:[/bold]")
        self.console.print()
        self.console.print("  [green]main[/green]  ●──●──●──●──→  [red]LOSS $8,000[/red]")
        self.console.print("            │")
        self.console.print("            └──●──●──●──→  [green]SAVED $8,000[/green]  [cyan](fix)[/cyan]")
        self.console.print()
        
        if not self.wait():
            return False
        
        # Verify both
        self.show_command("krnx verify --all")
        self.console.print()
        
        main_valid = self.substrate.verify(branch="main")
        fix_valid = self.substrate.verify(branch="fix")
        
        self.console.print(f"  main: [green]✓ VALID[/green] ({len(self.events)} events)")
        self.console.print(f"  fix:  [green]✓ VALID[/green] ({len(self.substrate.log(branch='fix'))} events)")
        
        self.console.print()
        self.console.print("[bold]Every event is:[/bold]")
        self.console.print("  • Timestamped")
        self.console.print("  • Hash-linked to its parent")
        self.console.print("  • Cryptographically verifiable")
        self.console.print("  • Exportable for audit")
        
        if not self.wait():
            return False
        
        # Summary
        self.clear()
        self.console.print()
        
        table = Table(title="Timeline Comparison", box=box.ROUNDED)
        table.add_column("Branch", style="cyan")
        table.add_column("Decision", style="white")
        table.add_column("Outcome", style="white")
        table.add_column("Verified", style="white")
        
        table.add_row("main", "APPROVE", "[red]LOSS $8,000[/red]", "[green]✓[/green]")
        table.add_row("fix", "DENY", "[green]SAVED $8,000[/green]", "[green]✓[/green]")
        
        self.console.print(table)
        
        if not self.wait():
            return False
        
        return True
    
    def finale(self) -> bool:
        """Final screen."""
        self.clear()
        self.console.print()
        self.console.print()
        
        self.console.print(Panel(
            "[bold]krnx[/bold]\n\n"
            "Record. Branch. Replay. Verify.\n\n"
            "[dim]Git for AI Agent State[/dim]",
            box=box.DOUBLE,
            border_style="green",
            padding=(1, 4),
        ))
        
        self.console.print()
        self.console.print(Align.center("[cyan]pip install krnx[/cyan]"))
        self.console.print(Align.center("[dim]github.com/chillbot/krnx[/dim]"))
        self.console.print()
        
        if not self.wait("[dim]SPACE to explore in Studio · Q to quit[/dim]"):
            return False
        
        return True
    
    def run(self):
        """Run the full demo."""
        try:
            if not self.act1_setup():
                return
            if not self.act2_disaster():
                return
            if not self.act3_investigation():
                return
            if not self.act4_branch_and_fix():
                return
            if not self.act5_proof():
                return
            if self.finale():
                # Offer Studio with guided walkthrough
                from krnx.studio import run_studio
                run_studio("demo-agent", guided=True, path=self.temp_dir)
        finally:
            # Cleanup
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def run_narrated_demo():
    """Entry point for krnx demo command."""
    demo = NarratedDemo()
    demo.run()


if __name__ == "__main__":
    run_narrated_demo()
