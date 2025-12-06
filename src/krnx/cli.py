"""
krnx CLI — Git for ML Agent State

Usage:
    krnx init <name>
    krnx record <type> <json>
    krnx log
    krnx show <event-id>
    krnx branch <name>
    krnx diff <a>..<b>
    krnx verify
    krnx studio
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint

from . import Substrate, init as krnx_init, __version__

# =============================================================================
# APP SETUP
# =============================================================================

app = typer.Typer(
    name="krnx",
    help="Git for ML agent state",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

# Config file for current workspace
CONFIG_DIR = Path(".krnx")
CONFIG_FILE = CONFIG_DIR / "config.json"


def get_config() -> dict:
    """Load config."""
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {}


def save_config(config: dict):
    """Save config."""
    CONFIG_DIR.mkdir(exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def get_substrate() -> Substrate:
    """Get substrate from config or error."""
    config = get_config()
    if "workspace" not in config:
        console.print("[red]No workspace. Run 'krnx init <name>' first.[/red]")
        raise typer.Exit(1)
    return krnx_init(
        name=config["workspace"],
        path=config.get("path", ".krnx"),
    )


def get_current_branch() -> str:
    """Get current branch from config."""
    config = get_config()
    return config.get("branch", "main")


def set_current_branch(branch: str):
    """Set current branch in config."""
    config = get_config()
    config["branch"] = branch
    save_config(config)


def format_ts(ts: float) -> str:
    """Format timestamp for display."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# COMMANDS
# =============================================================================

@app.command()
def init(
    name: str = typer.Argument(..., help="Workspace name"),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="Storage path"),
):
    """Create or open a workspace."""
    storage_path = path or ".krnx"
    s = krnx_init(name=name, path=storage_path)
    
    # Save config
    save_config({
        "workspace": name,
        "path": storage_path,
        "branch": "main",
    })
    
    console.print(f"[green]✓[/green] Initialized workspace '{name}' at {s.db_path}")


@app.command()
def record(
    type: str = typer.Argument(..., help="Event type (think, observe, act, result, etc)"),
    content: str = typer.Argument(..., help="JSON content"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Agent name"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Branch name"),
):
    """Record an event."""
    s = get_substrate()
    branch = branch or get_current_branch()
    
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        console.print(f"[red]Invalid JSON: {content}[/red]")
        raise typer.Exit(1)
    
    kwargs = {"type": type, "content": data, "branch": branch}
    if agent:
        kwargs["agent"] = agent
    
    event_id = s.record(**kwargs)
    console.print(f"[green]✓[/green] Recorded {event_id}")


@app.command("log")
def log_cmd(
    limit: int = typer.Option(20, "--limit", "-n", help="Max events"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Branch"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Filter by agent"),
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by type"),
    before: Optional[float] = typer.Option(None, "--before", help="Before timestamp"),
    after: Optional[float] = typer.Option(None, "--after", help="After timestamp"),
):
    """Show timeline."""
    s = get_substrate()
    branch = branch or get_current_branch()
    
    events = s.log(
        limit=limit,
        branch=branch,
        agent=agent,
        type=type,
        before=before,
        after=after,
    )
    
    if not events:
        console.print("[dim]No events[/dim]")
        return
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="cyan", width=20)
    table.add_column("Type", style="yellow", width=10)
    table.add_column("Agent", style="magenta", width=12)
    table.add_column("Time", style="dim", width=19)
    table.add_column("Content", overflow="ellipsis")
    
    for e in events:
        content_str = json.dumps(e.content)
        if len(content_str) > 50:
            content_str = content_str[:47] + "..."
        table.add_row(
            e.id,
            e.type,
            e.agent,
            format_ts(e.ts),
            content_str,
        )
    
    console.print(table)
    console.print(f"[dim]Branch: {branch} | Events: {len(events)}[/dim]")


@app.command()
def show(
    event_id: str = typer.Argument(..., help="Event ID"),
):
    """Inspect a single event."""
    s = get_substrate()
    event = s.show(event_id)
    
    if not event:
        console.print(f"[red]Event not found: {event_id}[/red]")
        raise typer.Exit(1)
    
    console.print(Panel(
        f"[bold]ID:[/bold]      {event.id}\n"
        f"[bold]Type:[/bold]    {event.type}\n"
        f"[bold]Agent:[/bold]   {event.agent}\n"
        f"[bold]Branch:[/bold]  {event.branch}\n"
        f"[bold]Time:[/bold]    {format_ts(event.ts)}\n"
        f"[bold]Hash:[/bold]    {event.hash}\n"
        f"[bold]Parent:[/bold]  {event.parent or '(none)'}\n"
        f"\n[bold]Content:[/bold]",
        title="Event",
    ))
    console.print(Syntax(json.dumps(event.content, indent=2), "json"))


@app.command()
def at(
    ref: str = typer.Argument(..., help="Timestamp or event ID"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Branch"),
):
    """Show state at a point in time."""
    s = get_substrate()
    branch = branch or get_current_branch()
    
    # Try to parse as timestamp
    try:
        ref_val = float(ref)
    except ValueError:
        ref_val = ref  # Event ID
    
    events = s.at(ref_val, branch=branch)
    
    if not events:
        console.print("[dim]No events at that point[/dim]")
        return
    
    console.print(f"[bold]State at {ref}:[/bold] {len(events)} events\n")
    
    for e in events[-10:]:  # Show last 10
        content_str = json.dumps(e.content)
        if len(content_str) > 60:
            content_str = content_str[:57] + "..."
        console.print(f"  [cyan]{e.id[:16]}[/cyan] [yellow]{e.type:10}[/yellow] {content_str}")
    
    if len(events) > 10:
        console.print(f"  [dim]... and {len(events) - 10} more[/dim]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search string"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Branch"),
):
    """Search events by content."""
    s = get_substrate()
    branch = branch or get_current_branch()
    
    events = s.search(query, limit=limit, branch=branch)
    
    if not events:
        console.print(f"[dim]No events matching '{query}'[/dim]")
        return
    
    console.print(f"[bold]Found {len(events)} events:[/bold]\n")
    
    for e in events:
        content_str = json.dumps(e.content)
        if len(content_str) > 60:
            content_str = content_str[:57] + "..."
        console.print(f"  [cyan]{e.id}[/cyan] [yellow]{e.type:10}[/yellow] {content_str}")


@app.command("branch")
def branch_cmd(
    name: str = typer.Argument(..., help="Branch name"),
    from_event: Optional[str] = typer.Option(None, "--from", "-f", help="Fork from event ID"),
):
    """Create a new branch."""
    s = get_substrate()
    
    try:
        s.branch(name, from_event=from_event)
        console.print(f"[green]✓[/green] Created branch '{name}'")
        if from_event:
            console.print(f"  [dim]Forked from {from_event}[/dim]")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command("branch-delete")
def branch_delete_cmd(
    name: str = typer.Argument(..., help="Branch name"),
):
    """Delete a branch (soft delete)."""
    s = get_substrate()
    
    try:
        s.branch_delete(name)
        console.print(f"[green]✓[/green] Deleted branch '{name}'")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command()
def branches(
    deleted: bool = typer.Option(False, "--deleted", "-d", help="Show deleted branches"),
):
    """List all branches."""
    s = get_substrate()
    current = get_current_branch()
    
    branch_list = s.branches(deleted=deleted)
    
    if not branch_list:
        console.print("[dim]No branches[/dim]")
        return
    
    for b in branch_list:
        marker = "→ " if b["name"] == current else "  "
        deleted_marker = " [dim](deleted)[/dim]" if b.get("deleted_at") else ""
        fork_info = f" [dim]from {b['fork_event_id'][:12]}...[/dim]" if b.get("fork_event_id") else ""
        
        console.print(f"{marker}[bold]{b['name']}[/bold]{fork_info}{deleted_marker}")


@app.command()
def checkout(
    branch: str = typer.Argument(..., help="Branch name"),
):
    """Switch to a branch."""
    s = get_substrate()
    
    # Verify branch exists
    branch_list = s.branches()
    branch_names = [b["name"] for b in branch_list]
    
    if branch not in branch_names:
        console.print(f"[red]Branch '{branch}' not found[/red]")
        raise typer.Exit(1)
    
    set_current_branch(branch)
    console.print(f"[green]✓[/green] Switched to '{branch}'")


@app.command()
def diff(
    spec: str = typer.Argument(..., help="Branch comparison (e.g., main..fix)"),
):
    """Compare two branches."""
    s = get_substrate()
    
    if ".." not in spec:
        console.print("[red]Usage: krnx diff branch_a..branch_b[/red]")
        raise typer.Exit(1)
    
    branch_a, branch_b = spec.split("..", 1)
    
    result = s.diff(branch_a, branch_b)
    
    console.print(f"\n[bold]Comparing {branch_a} → {branch_b}[/bold]\n")
    
    console.print(f"[green]Common:[/green] {len(result['common'])} events")
    console.print(f"[yellow]Only in {branch_a}:[/yellow] {len(result['only_a'])} events")
    console.print(f"[cyan]Only in {branch_b}:[/cyan] {len(result['only_b'])} events")
    
    if result["only_a"]:
        console.print(f"\n[yellow]Only in {branch_a}:[/yellow]")
        for e in result["only_a"][:5]:
            console.print(f"  - {e.id} {e.type}: {json.dumps(e.content)[:50]}")
        if len(result["only_a"]) > 5:
            console.print(f"  [dim]... and {len(result['only_a']) - 5} more[/dim]")
    
    if result["only_b"]:
        console.print(f"\n[cyan]Only in {branch_b}:[/cyan]")
        for e in result["only_b"][:5]:
            console.print(f"  + {e.id} {e.type}: {json.dumps(e.content)[:50]}")
        if len(result["only_b"]) > 5:
            console.print(f"  [dim]... and {len(result['only_b']) - 5} more[/dim]")


# =============================================================================
# CHECKPOINTS
# =============================================================================

@app.command("checkpoint")
def checkpoint_cmd(
    name: str = typer.Argument(..., help="Checkpoint name"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Description"),
    event: Optional[str] = typer.Option(None, "--event", "-e", help="Specific event ID"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Branch"),
):
    """Create a named checkpoint at the current position."""
    s = get_substrate()
    branch = branch or get_current_branch()
    
    try:
        event_id = s.checkpoint(
            name=name,
            description=description,
            branch=branch,
            event_id=event,
        )
        console.print(f"[green]✓[/green] Checkpoint '{name}' created at {event_id[:16]}...")
        if description:
            console.print(f"  [dim]{description}[/dim]")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command("checkpoint-delete")
def checkpoint_delete_cmd(
    name: str = typer.Argument(..., help="Checkpoint name"),
):
    """Delete a checkpoint."""
    s = get_substrate()
    
    try:
        s.checkpoint_delete(name)
        console.print(f"[green]✓[/green] Deleted checkpoint '{name}'")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command("checkpoints")
def checkpoints_cmd(
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Filter by branch"),
):
    """List all checkpoints."""
    s = get_substrate()
    
    checkpoint_list = s.checkpoints(branch=branch)
    
    if not checkpoint_list:
        console.print("[dim]No checkpoints[/dim]")
        return
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Branch", style="yellow")
    table.add_column("Event", style="dim", width=20)
    table.add_column("Created", style="dim")
    table.add_column("Description", overflow="ellipsis")
    
    for cp in checkpoint_list:
        table.add_row(
            cp["name"],
            cp["branch"],
            cp["event_id"][:16] + "..." if cp["event_id"] else "-",
            format_ts(cp["created_at"]) if cp["created_at"] else "-",
            cp["description"] or "",
        )
    
    console.print(table)


@app.command("branch-from-checkpoint")
def branch_from_checkpoint_cmd(
    branch_name: str = typer.Argument(..., help="New branch name"),
    checkpoint_name: str = typer.Argument(..., help="Checkpoint to branch from"),
):
    """Create a branch from a checkpoint."""
    s = get_substrate()
    
    try:
        s.branch_from_checkpoint(branch_name, checkpoint_name)
        cp = s.get_checkpoint(checkpoint_name)
        console.print(f"[green]✓[/green] Created branch '{branch_name}' from checkpoint '{checkpoint_name}'")
        console.print(f"  [dim]At event {cp['event_id'][:16]}...[/dim]")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command()
def verify(
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Branch to verify"),
):
    """Verify hash chain integrity."""
    s = get_substrate()
    branch = branch or get_current_branch()
    
    try:
        s.verify(branch)
        count = s.count(branch)
        console.print(f"[green]✓[/green] Branch '{branch}' intact ({count} events)")
    except Exception as e:
        console.print(f"[red]✗ Integrity error:[/red] {e}")
        raise typer.Exit(1)


# =============================================================================
# OBSERVABILITY
# =============================================================================

@app.command()
def stats(
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Filter by branch"),
    since: Optional[float] = typer.Option(None, "--since", "-s", help="Hours to look back"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Show workspace statistics."""
    s = get_substrate()
    
    data = s.stats(branch=branch, since_hours=since)
    
    if output_json:
        print(json.dumps(data, indent=2))
        return
    
    # Header
    console.print()
    console.print(Panel(
        f"[bold]Events:[/bold] {data['total_events']:,}  │  "
        f"[bold]Branches:[/bold] {len(data['branches'])}  │  "
        f"[bold]Checkpoints:[/bold] {data['checkpoints']}  │  "
        f"[bold]Tokens:[/bold] {data['tokens_total']:,}",
        title="[bold cyan]krnx stats[/bold cyan]",
        border_style="cyan",
    ))
    
    # Recent activity
    console.print(f"\n[dim]Last 24h:[/dim] {data['recent_24h']} events")
    
    # By type
    if data["by_type"]:
        console.print("\n[bold]By Type:[/bold]")
        total = sum(data["by_type"].values())
        for t, count in list(data["by_type"].items())[:8]:
            pct = (count / total) * 100 if total > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            console.print(f"  [yellow]{t:12}[/yellow] {bar} {count:>5} ({pct:4.1f}%)")
    
    # By agent
    if data["by_agent"]:
        console.print("\n[bold]By Agent:[/bold]")
        total = sum(data["by_agent"].values())
        for agent, count in list(data["by_agent"].items())[:5]:
            pct = (count / total) * 100 if total > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            console.print(f"  [magenta]{agent:12}[/magenta] {bar} {count:>5} ({pct:4.1f}%)")
    
    # By branch
    if data["by_branch"] and len(data["by_branch"]) > 1:
        console.print("\n[bold]By Branch:[/bold]")
        for br, count in list(data["by_branch"].items())[:5]:
            console.print(f"  [cyan]{br:12}[/cyan] {count:>5} events")
    
    # Time range
    if data["time_range"]:
        start = format_ts(data["time_range"]["start"])
        end = format_ts(data["time_range"]["end"])
        console.print(f"\n[dim]Time range: {start} → {end}[/dim]")
    
    # Cost estimate (rough: $0.003 per 1K tokens for GPT-4)
    if data["tokens_total"] > 0:
        cost_estimate = (data["tokens_total"] / 1000) * 0.003
        console.print(f"[dim]Est. cost: ~${cost_estimate:.2f} (at $0.003/1K tokens)[/dim]")
    
    console.print()


@app.command("export")
def export_cmd(
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Branch to export"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
):
    """Export branch to JSONL."""
    s = get_substrate()
    branch = branch or get_current_branch()
    
    if output:
        s.export(branch, output)
        console.print(f"[green]✓[/green] Exported to {output}")
    else:
        print(s.export(branch))


@app.command("import")
def import_cmd(
    path: str = typer.Argument(..., help="JSONL file to import"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Target branch"),
):
    """Import events from JSONL."""
    s = get_substrate()
    branch = branch or get_current_branch()
    
    count = s.import_events(path, branch)
    console.print(f"[green]✓[/green] Imported {count} events to '{branch}'")


@app.command()
def studio():
    """Launch Krnx Studio (TUI)."""
    try:
        from .studio import run_studio
        config = get_config()
        workspace = config.get("workspace")
        run_studio(workspace)
    except ImportError:
        console.print("[red]Studio requires textual. Install with:[/red]")
        console.print("  pip install krnx[studio]")
        raise typer.Exit(1)


@app.command()
def demo():
    """Run the interactive krnx story demo (no API key needed)."""
    try:
        from .narrated_demo import run_narrated_demo
        run_narrated_demo()
    except ImportError as e:
        console.print(f"[red]Demo requires rich. Install with:[/red]")
        console.print("  pip install rich")
        raise typer.Exit(1)


@app.command(name="try")
def try_demo():
    """
    Run live demo with real LLM calls (requires OPENAI_API_KEY or ANTHROPIC_API_KEY).
    
    This runs a customer service agent that:
    1. Processes a refund request (makes a mistake)
    2. Branches to try with fraud-check context
    3. Makes correct decision on the branch
    4. Verifies both timelines
    
    Set your API key first:
        export ANTHROPIC_API_KEY=sk-...
    """
    try:
        from .cs_agent import run_try_demo
        result = run_try_demo()
        raise typer.Exit(result)
    except ImportError as e:
        if "anthropic" in str(e):
            console.print("[red]Try demo requires anthropic. Install with:[/red]")
            console.print("  pip install anthropic")
        else:
            console.print(f"[red]Import error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version."""
    console.print(f"krnx {__version__}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    app()


if __name__ == "__main__":
    main()
