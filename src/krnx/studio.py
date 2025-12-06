"""
Krnx Studio — Visual Timeline Explorer

Features:
- Guided walkthrough mode (from demo)
- Visual branch/timeline graph
- Event inspection with hash chain context
- Branch comparison
- Clear navigation

Usage:
    krnx studio [workspace]
    
Controls:
    ↑↓/j/k   Navigate events
    ←→/h/l   Switch branches  
    Enter    Inspect event (full details)
    t        Toggle timeline view
    v        Verify hash chain
    b        Create branch
    d        Diff branches
    ?        Toggle help
    q        Back/Quit
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from textual.app import App, ComposeResult
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Static, Header, Footer, ListView, ListItem, 
    Label, RichLog, Button, Input, DataTable
)
from textual.containers import Horizontal, Vertical, Container, ScrollableContainer, Grid
from textual.binding import Binding
from textual.message import Message
from textual import events
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

from . import Substrate, init as krnx_init, Event


# =============================================================================
# HELPERS
# =============================================================================

def format_ts(ts: float) -> str:
    """Short timestamp format."""
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

def format_ts_full(ts: float) -> str:
    """Full timestamp format."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def find_workspaces(path: str = ".krnx") -> List[str]:
    """Find existing workspaces."""
    krnx_dir = Path(path)
    if not krnx_dir.exists():
        return []
    workspaces = []
    for f in krnx_dir.iterdir():
        if f.suffix == ".db":
            workspaces.append(f.stem)
    return sorted(workspaces)


# =============================================================================
# STYLES  
# =============================================================================

CSS = """
Screen {
    background: #0f1612;
}

Header {
    background: #1a2420;
    color: #6bbd8a;
}

Footer {
    background: #1a2420;
    color: #6bbd8a;
}

/* Main layout */
#main-container {
    height: 1fr;
}

/* Branch bar */
#branch-bar {
    height: 3;
    background: #1a2420;
    padding: 0 2;
    border-bottom: solid #2a3a30;
}

#branch-selector {
    width: 1fr;
}

#branch-info {
    color: #6bbd8a;
    text-style: bold;
}

/* Timeline graph panel - BIGGER */
#timeline-panel {
    height: 14;
    border-bottom: solid #2a3a30;
    padding: 1 2;
    background: #0d1410;
}

#timeline-content {
    height: 100%;
    scrollbar-size: 0 0;
}

/* Main split */
#content-split {
    height: 1fr;
}

/* Event list panel - WIDER (2fr) */
#event-panel {
    width: 2fr;
    min-width: 60;
    border: solid #2a3a30;
    padding: 0 1;
}

#event-panel.focused {
    border: solid #6bbd8a;
}

#event-panel-title {
    color: #5a7a6a;
    text-style: bold;
    padding: 0 1;
    height: 2;
}

#event-panel.focused #event-panel-title {
    color: #6bbd8a;
}

#event-list {
    height: 1fr;
    scrollbar-size: 1 1;
}

/* Detail panel - NARROWER (1fr) */
#detail-panel {
    width: 1fr;
    min-width: 35;
    padding: 1 2;
    border: solid #2a3a30;
}

#detail-title {
    text-style: bold;
    color: #6bbd8a;
    padding-bottom: 1;
}

#detail-content {
    height: 1fr;
}

/* Hash chain panel - navigable */
#chain-panel {
    height: 6;
    border: solid #2a3a30;
    padding: 0 2;
    background: #0d1410;
}

#chain-panel.focused {
    border: solid #6bbd8a;
}

#chain-title {
    color: #5a7a6a;
    text-style: bold;
}

#chain-panel.focused #chain-title {
    color: #6bbd8a;
}

#chain-content {
    height: 1fr;
}

/* Timeline panel - navigable, TALLER */
#timeline-panel {
    height: 18;
    border: solid #2a3a30;
    padding: 1 2;
    background: #0d1410;
}

#timeline-panel.focused {
    border: solid #6bbd8a;
}

/* Status bar */
#status-bar {
    height: 1;
    background: #141c17;
    color: #7a9a8a;
    padding: 0 2;
}

/* Event item styling - with wrapping */
.event-item {
    padding: 0 1;
    height: auto;
    min-height: 2;
}

.event-item:hover {
    background: #1a2a20;
}

.event-item:focus {
    background: #2a3a30;
}

.event-type {
    color: #6bbd8a;
    text-style: bold;
    width: 10;
}

.event-hash {
    color: #4a6a5a;
    width: 10;
}

.event-time {
    color: #5a7a6a;
}

.event-preview {
    color: #8aaa9a;
    width: 1fr;
}

/* Guide overlay */
#guide-overlay {
    align: center middle;
    background: rgba(0, 0, 0, 0.85);
}

#guide-box {
    width: 70;
    height: auto;
    max-height: 35;
    background: #141c17;
    border: double #3a5a4a;
    padding: 2 3;
}

#guide-title {
    text-style: bold;
    color: #6bbd8a;
    text-align: center;
    padding-bottom: 1;
}

#guide-content {
    color: #aacaba;
    padding: 1 0;
}

#guide-hint {
    text-align: center;
    color: #6bbd8a;
    padding-top: 1;
    text-style: bold;
}

/* Menu styling */
#main-menu {
    align: center middle;
}

#menu-box {
    width: 60;
    height: auto;
    border: solid #2a3a30;
    padding: 1 2;
    background: #141c17;
}

#menu-title {
    text-align: center;
    text-style: bold;
    color: #6bbd8a;
    padding-bottom: 1;
}

.workspace-item {
    padding: 0 1;
    height: 3;
}

.workspace-item:hover {
    background: #1a2a20;
}

.workspace-item:focus {
    background: #2a3a30;
}

/* Inspector */
#inspector {
    padding: 1 2;
}

.field-row {
    height: 2;
}

.field-label {
    width: 12;
    color: #5a7a6a;
}

.field-value {
    color: #aacaba;
}

.field-value-hash {
    color: #6bbd8a;
    text-style: bold;
}

#content-box {
    border: solid #2a3a30;
    padding: 1;
    margin-top: 1;
    background: #0d1410;
    height: auto;
    min-height: 10;
}

/* DIFF SCREEN - Fixed layout */
#diff-container {
    height: 1fr;
    width: 100%;
}

#diff-left {
    width: 1fr;
    height: 100%;
    border-right: solid #2a3a30;
    padding: 1 2;
}

#diff-right {
    width: 1fr;
    height: 100%;
    padding: 1 2;
}

.diff-title {
    text-style: bold;
    text-align: center;
    padding-bottom: 1;
    height: 2;
}

#diff-left-title {
    color: #6bbd8a;
}

#diff-right-title {
    color: #c4956a;
}

#diff-left-content {
    height: 1fr;
}

#diff-right-content {
    height: 1fr;
}
"""


# =============================================================================
# GUIDE STEPS (for guided mode)
# =============================================================================

GUIDE_STEPS = [
    {
        "title": "Welcome to krnx Studio",
        "content": (
            "This is where you explore your agent's timeline.\n\n"
            "You just saw your agent make a mistake — approve a\n"
            "fraudulent refund. Then you branched, replayed with\n"
            "fraud detection, and it denied.\n\n"
            "Now let's explore both timelines."
        ),
    },
    {
        "title": "The Timeline Graph",
        "content": (
            "At the top, you see the branch visualization.\n\n"
            "Each [bold]●[/bold] is an event. The hash beneath it\n"
            "identifies it uniquely.\n\n"
            "  [green]main[/]   ●──●──●──●   (original timeline)\n"
            "                └──●──●──●   [yellow]fix[/] (alternate)\n\n"
            "The 'fix' branch forks from the first event."
        ),
    },
    {
        "title": "Focus & Navigation",
        "content": (
            "The UI has three navigable panes:\n\n"
            "  • [bold]Events[/]   — The event list\n"
            "  • [bold]Chain[/]    — The hash chain\n"
            "  • [bold]Timeline[/] — The branch graph\n\n"
            "Press [bold]Tab[/] to cycle focus between them.\n"
            "The focused pane has a [green]green border[/]."
        ),
    },
    {
        "title": "Walking the Chain",
        "content": (
            "The hash chain shows event links:\n\n"
            "  genesis → [green]a7f3[/] → [green]b82c[/] → [green]9d1e[/]\n\n"
            "Navigate it with:\n"
            "  • [bold]Tab[/] to focus the chain pane\n"
            "  • [bold]← →[/] to move through hashes\n"
            "  • [bold]Enter[/] to jump to that event\n\n"
            "Or use [bold][[/] and [bold]][/] from anywhere!"
        ),
    },
    {
        "title": "Event Details",
        "content": (
            "On the right, the selected event's details.\n\n"
            "Notice the [bold green]Hash[/] and [bold]Parent[/] fields.\n"
            "Every event links to its parent via hash.\n\n"
            "This is the tamper-evident chain — if anyone\n"
            "modifies an event, the chain breaks."
        ),
    },
    {
        "title": "Switching Branches",
        "content": (
            "With Events or Timeline focused:\n"
            "  • [bold]← →[/] switches between branches\n\n"
            "With Chain focused:\n"
            "  • [bold]← →[/] walks the chain\n\n"
            "Press [bold]d[/] to diff two branches side-by-side."
        ),
    },
    {
        "title": "You're Ready!",
        "content": (
            "Key commands:\n\n"
            "  [bold]Tab[/]      Cycle focus (Events/Chain/Timeline)\n"
            "  [bold]↑↓[/]       Navigate in focused pane\n"
            "  [bold]← →[/]      Move in chain / switch branch\n"
            "  [bold][ ][/]      Walk chain (global shortcut)\n"
            "  [bold]Enter[/]    Select / Inspect\n"
            "  [bold]v[/]        Verify chain integrity\n"
            "  [bold]d[/]        Diff branches\n"
            "  [bold]?[/]        Show this help\n"
            "  [bold]q[/]        Quit"
        ),
    },
]


# =============================================================================
# CUSTOM WIDGETS
# =============================================================================

class EventListItem(ListItem):
    """Custom event list item with rich display."""
    
    def __init__(self, event: Event, is_selected: bool = False):
        super().__init__(id=f"evt-{event.id}")
        self.event = event
        self.is_selected = is_selected
    
    def compose(self) -> ComposeResult:
        e = self.event
        
        # Type with color
        type_color = {
            "observe": "#6bbd8a",
            "think": "#7cb7d4", 
            "act": "#d4c47c",
            "result": "#c47cd4",
        }.get(e.type, "#8a9a8a")
        
        # Content preview - longer for wider panel
        content_str = str(e.content)
        if len(content_str) > 80:
            content_str = content_str[:80] + "..."
        
        with Horizontal(classes="event-item"):
            yield Static(f"[{type_color}]{e.type:8}[/]", classes="event-type")
            yield Static(f"[#5a8a6a]{e.hash[:8]}[/]", classes="event-hash")
            yield Static(content_str, classes="event-preview")


# =============================================================================
# GUIDE OVERLAY - Fixed navigation
# =============================================================================

class GuideOverlay(ModalScreen):
    """Guided walkthrough overlay."""
    
    BINDINGS = [
        Binding("space", "next", "Continue", show=True),
        Binding("enter", "next", "Continue", show=False),
        Binding("escape", "dismiss", "Skip", show=True),
        Binding("s", "dismiss", "Skip", show=False),
        Binding("q", "dismiss", "Skip", show=False),
    ]
    
    def __init__(self, step_index: int = 0):
        super().__init__()
        self.step_index = step_index
    
    def compose(self) -> ComposeResult:
        step = GUIDE_STEPS[self.step_index]
        total = len(GUIDE_STEPS)
        
        with Container(id="guide-overlay"):
            with Vertical(id="guide-box"):
                yield Static(f"[bold]{step['title']}[/] ({self.step_index + 1}/{total})", id="guide-title")
                yield Static(step["content"], id="guide-content")
                yield Static("[SPACE] Continue    [ESC] Skip guide", id="guide-hint")
    
    def action_next(self):
        """Go to next step or close."""
        if self.step_index < len(GUIDE_STEPS) - 1:
            self.app.pop_screen()
            self.app.push_screen(GuideOverlay(self.step_index + 1))
        else:
            self.app.pop_screen()
    
    def action_dismiss(self):
        """Close guide."""
        self.app.pop_screen()


# =============================================================================
# MAIN MENU SCREEN
# =============================================================================

class MainMenuScreen(Screen):
    """Workspace selection screen."""
    
    BINDINGS = [
        Binding("enter", "select", "Open"),
        Binding("n", "new_workspace", "New"),
        Binding("q", "quit", "Quit"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main-menu"):
            with Vertical(id="menu-box"):
                yield Static("KRNX STUDIO", id="menu-title")
                yield Static("─" * 40, classes="dim")
                yield ListView(id="workspace-list")
                yield Static("")
                yield Static("[N] New  [Enter] Open  [Q] Quit", classes="dim")
        yield Footer()
    
    def on_mount(self):
        self.refresh_workspaces()
    
    def refresh_workspaces(self):
        """Refresh workspace list."""
        list_view = self.query_one("#workspace-list", ListView)
        list_view.clear()
        
        workspaces = find_workspaces()
        
        if not workspaces:
            list_view.append(ListItem(Label("[dim]No workspaces found[/dim]")))
            list_view.append(ListItem(Label("[dim]Run 'krnx init <n>' first[/dim]")))
        else:
            for ws in workspaces:
                try:
                    s = krnx_init(ws)
                    count = s.count()
                    branches = len(s.branches())
                    list_view.append(ListItem(
                        Label(f"  {ws:<20} {count:>5} events  {branches:>2} branches"),
                        id=f"ws-{ws}"
                    ))
                except Exception:
                    list_view.append(ListItem(
                        Label(f"  {ws:<20} [error]"),
                        id=f"ws-{ws}"
                    ))
    
    def action_select(self):
        """Open selected workspace."""
        list_view = self.query_one("#workspace-list", ListView)
        if list_view.highlighted_child:
            item_id = list_view.highlighted_child.id
            if item_id and item_id.startswith("ws-"):
                workspace = item_id[3:]
                self.app.open_workspace(workspace)
    
    def action_new_workspace(self):
        """Create new workspace."""
        self.app.push_screen(NewWorkspaceScreen())
    
    def action_quit(self):
        self.app.exit()


class NewWorkspaceScreen(Screen):
    """Create new workspace screen."""
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main-menu"):
            with Vertical(id="menu-box"):
                yield Static("NEW WORKSPACE", id="menu-title")
                yield Static("─" * 40, classes="dim")
                yield Static("Name:")
                yield Input(placeholder="my-agent", id="ws-name")
                yield Static("")
                yield Button("Create", id="create-btn", variant="primary")
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "create-btn":
            name_input = self.query_one("#ws-name", Input)
            name = name_input.value.strip()
            if name:
                try:
                    krnx_init(name)
                    self.app.pop_screen()
                    self.app.open_workspace(name)
                except Exception as e:
                    self.notify(f"Error: {e}", severity="error")
    
    def action_cancel(self):
        self.app.pop_screen()


# =============================================================================
# TIMELINE SCREEN (main view)
# =============================================================================

class TimelineScreen(Screen):
    """Main timeline view with event list, details, and hash chain."""
    
    # Focus states
    FOCUS_EVENTS = "events"
    FOCUS_CHAIN = "chain"
    FOCUS_TIMELINE = "timeline"
    
    BINDINGS = [
        Binding("q", "back", "Quit"),
        Binding("escape", "back", "Back"),
        Binding("enter", "select", "Select"),
        Binding("up", "move_up", "↑", show=False),
        Binding("down", "move_down", "↓", show=False),
        Binding("k", "move_up", "↑", show=False),
        Binding("j", "move_down", "↓", show=False),
        Binding("left", "move_left", "←"),
        Binding("right", "move_right", "→"),
        Binding("h", "move_left", "←", show=False),
        Binding("l", "move_right", "→", show=False),
        Binding("tab", "cycle_focus", "Tab"),
        Binding("shift+tab", "cycle_focus_back", "Shift+Tab", show=False),
        Binding("left_square_bracket", "chain_back", "[", show=False),  # [ key
        Binding("right_square_bracket", "chain_forward", "]", show=False),  # ] key
        Binding("comma", "chain_back", "<", show=False),  # < key (alternative)
        Binding("full_stop", "chain_forward", ">", show=False),  # > key (alternative)
        Binding("v", "verify", "Verify"),
        Binding("b", "create_branch", "Branch"),
        Binding("d", "diff", "Diff"),
        Binding("question_mark", "show_guide", "Help"),
    ]
    
    def __init__(self, workspace: str, guided: bool = False, path: Optional[str] = None):
        super().__init__()
        self.workspace = workspace
        self.guided = guided
        self.path = path
        if path:
            self.substrate = krnx_init(workspace, path=path)
        else:
            self.substrate = krnx_init(workspace)
        self.current_branch = "main"
        self.branches_list = []
        self.events_list: List[Event] = []
        self.events_chrono: List[Event] = []  # Chronological order for chain
        self.selected_index = 0
        self.chain_index = 0  # Position in chain (chronological)
        self.current_focus = self.FOCUS_EVENTS  # Start with events focused
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Vertical(id="main-container"):
            # Branch bar
            with Horizontal(id="branch-bar"):
                yield Static(f"⎇ {self.current_branch}", id="branch-info")
                yield Static(f"  │  {self.workspace}  │  [dim]Tab[/dim] cycle panes  [dim][][/dim] walk chain", id="branch-selector")
            
            # Timeline graph
            with Container(id="timeline-panel"):
                yield RichLog(id="timeline-content", markup=True, wrap=False)
            
            # Main content split
            with Horizontal(id="content-split"):
                # Event list
                with Vertical(id="event-panel", classes="focused"):
                    yield Static("EVENTS", id="event-panel-title")
                    yield ListView(id="event-list")
                
                # Detail panel
                with Vertical(id="detail-panel"):
                    yield Static("DETAILS", id="detail-title")
                    yield RichLog(id="detail-content", markup=True, wrap=True)
            
            # Hash chain panel
            with Vertical(id="chain-panel"):
                yield Static("CHAIN  [dim]← → navigate  Enter jump[/dim]", id="chain-title")
                yield RichLog(id="chain-content", markup=True, wrap=False)
        
        yield Static("", id="status-bar")
        yield Footer()
    
    def on_mount(self):
        """Initialize the view."""
        self.refresh_branches()
        self.refresh_events()
        self.refresh_timeline()
        self.refresh_chain()
        self.update_status()
        self._update_focus_visuals()
        
        # Show guide if in guided mode
        if self.guided:
            self.app.push_screen(GuideOverlay(0))
    
    def _update_focus_visuals(self):
        """Update pane borders based on focus."""
        event_panel = self.query_one("#event-panel")
        chain_panel = self.query_one("#chain-panel")
        timeline_panel = self.query_one("#timeline-panel")
        
        # Remove all focused classes
        event_panel.remove_class("focused")
        chain_panel.remove_class("focused")
        timeline_panel.remove_class("focused")
        
        # Add focused class to current
        if self.current_focus == self.FOCUS_EVENTS:
            event_panel.add_class("focused")
        elif self.current_focus == self.FOCUS_CHAIN:
            chain_panel.add_class("focused")
        elif self.current_focus == self.FOCUS_TIMELINE:
            timeline_panel.add_class("focused")
        
        self.update_status()
    
    def refresh_branches(self):
        """Get list of branches."""
        self.branches_list = [b["name"] for b in self.substrate.branches()]
        if self.current_branch not in self.branches_list and self.branches_list:
            self.current_branch = self.branches_list[0]
    
    def refresh_events(self):
        """Refresh event list for current branch."""
        list_view = self.query_one("#event-list", ListView)
        list_view.clear()
        
        self.events_list = self.substrate.log(limit=100, branch=self.current_branch)
        self.events_chrono = list(reversed(self.events_list))  # Oldest first
        
        if not self.events_list:
            list_view.append(ListItem(Label("[dim]No events on this branch[/dim]")))
            list_view.append(ListItem(Label("[dim]Record events with: krnx record <type> <json>[/dim]")))
        else:
            for i, event in enumerate(self.events_list):
                item = EventListItem(event, is_selected=(i == self.selected_index))
                list_view.append(item)
            
            # Select first item
            if self.events_list:
                self.selected_index = 0
                self.chain_index = len(self.events_chrono) - 1  # Most recent in chain
                self.refresh_detail()
        
        # Update branch info
        self.query_one("#branch-info", Static).update(f"⎇ [bold]{self.current_branch}[/bold]")
    
    def refresh_detail(self):
        """Refresh detail panel for selected event."""
        detail = self.query_one("#detail-content", RichLog)
        detail.clear()
        
        if not self.events_list or self.selected_index >= len(self.events_list):
            detail.write("[dim]Select an event[/dim]")
            return
        
        e = self.events_list[self.selected_index]
        
        # Compact header info
        detail.write(f"[bold]Type:[/] [{self._type_color(e.type)}]{e.type}[/]")
        detail.write(f"[bold]Time:[/] {format_ts_full(e.ts)}")
        detail.write("")
        detail.write(f"[bold]Hash:[/]")
        detail.write(f"  [green]{e.hash}[/]")
        detail.write(f"[bold]Parent:[/]")
        detail.write(f"  [dim]{e.parent or '(genesis)'}[/dim]")
        detail.write("")
        detail.write("[bold]Content:[/]")
        
        # Pretty print content
        content_str = json.dumps(e.content, indent=2)
        for line in content_str.split("\n"):
            detail.write(f"  {line}")
    
    def refresh_timeline(self):
        """Refresh the timeline graph - BIG and full width."""
        timeline = self.query_one("#timeline-content", RichLog)
        timeline.clear()
        
        branches = self.substrate.branches()
        main_events = self.substrate.log(limit=20, branch="main")
        main_events.reverse()
        
        if not main_events:
            timeline.write("")
            timeline.write("  [dim]No events yet — record some with:[/dim]")
            timeline.write("  [dim]krnx record observe '{\"key\": \"value\"}'[/dim]")
            return
        
        # BIG timeline with wide spacing
        # Use ◉ for bigger markers, longer lines
        
        # Main branch - wide spacing
        main_line = "  [#6bbd8a bold]main[/]   "
        label_line = "           "  # Align with main_line
        
        for i, e in enumerate(main_events[-8:]):  # Show up to 8 events
            is_selected = (self.current_branch == "main" and 
                          self.events_list and 
                          self.selected_index < len(self.events_list) and
                          e.id == self.events_list[self.selected_index].id)
            
            # Bigger marker, wider spacing
            if is_selected:
                marker = "[bold white on #2a5a4a] ◉ [/]"
            else:
                marker = "[#6bbd8a]─◉─[/]"
            
            main_line += f"────{marker}────"
            
            # Hash label below, centered
            hash_short = e.hash[:6]
            label_line += f"   [#5a8a6a]{hash_short}[/]   "
        
        main_line += "────▶"
        
        timeline.write(main_line)
        timeline.write(label_line)
        
        # Other branches with fork visualization
        for b in branches:
            if b["name"] != "main":
                branch_events = self.substrate.log(limit=10, branch=b["name"])
                if branch_events:
                    branch_events.reverse()
                    timeline.write("")
                    
                    # Fork point - aligned under first event
                    fork_indent = "                 "  # Space to align with fork point
                    fork_line = f"{fork_indent}╰"
                    fork_label = f"{fork_indent} "
                    
                    for i, e in enumerate(branch_events[-6:]):  # Show up to 6 events
                        is_selected = (self.current_branch == b["name"] and 
                                      self.events_list and 
                                      self.selected_index < len(self.events_list) and
                                      e.id == self.events_list[self.selected_index].id)
                        
                        if is_selected:
                            marker = "[bold white on #4a3a2a] ◉ [/]"
                        else:
                            marker = "[#c4956a]─◉─[/]"
                        
                        fork_line += f"────{marker}────"
                        hash_short = e.hash[:6]
                        fork_label += f"   [#8a6a4a]{hash_short}[/]   "
                    
                    fork_line += f"────▶ [#c4956a bold]{b['name']}[/]"
                    
                    timeline.write(fork_line)
                    timeline.write(fork_label)
    
    def refresh_chain(self):
        """Refresh hash chain with selection highlight."""
        chain = self.query_one("#chain-content", RichLog)
        chain.clear()
        
        if not self.events_chrono:
            chain.write("[dim]No events[/dim]")
            return
        
        # Build chain visualization with selection
        chain_parts = []
        
        # Genesis
        if self.chain_index == -1:  # Before first event
            chain_parts.append("[bold white on #2a5a4a]genesis[/]")
        else:
            chain_parts.append("[dim]genesis[/dim]")
        
        # Events
        display_start = max(0, self.chain_index - 4)
        display_end = min(len(self.events_chrono), display_start + 10)
        
        if display_start > 0:
            chain_parts.append("[dim]...[/dim]")
        
        for i in range(display_start, display_end):
            e = self.events_chrono[i]
            hash_short = e.hash[:6]
            
            if i == self.chain_index:
                # Selected in chain
                chain_parts.append(f"[bold white on #2a5a4a]{hash_short}[/]")
            else:
                chain_parts.append(f"[green]{hash_short}[/]")
        
        if display_end < len(self.events_chrono):
            chain_parts.append("[dim]...[/dim]")
        
        chain_str = " → ".join(chain_parts)
        chain.write(chain_str)
        
        # Show selected event type
        if 0 <= self.chain_index < len(self.events_chrono):
            e = self.events_chrono[self.chain_index]
            chain.write(f"  [dim]↑ {e.type} ({e.hash[:8]})[/dim]")
    
    def update_status(self):
        """Update status bar with focus info."""
        count = len(self.events_list)
        branch_count = len(self.branches_list)
        
        focus_name = {
            self.FOCUS_EVENTS: "Events",
            self.FOCUS_CHAIN: "Chain", 
            self.FOCUS_TIMELINE: "Timeline"
        }.get(self.current_focus, "")
        
        self.query_one("#status-bar", Static).update(
            f"Focus: [bold]{focus_name}[/]  │  "
            f"Events: {count}  │  Branches: {branch_count}  │  "
            f"[dim]Tab[/] focus  [dim][][/] chain  [dim]v[/] verify  [dim]?[/] help"
        )
    
    def _type_color(self, event_type: str) -> str:
        return {
            "observe": "#6bbd8a",
            "think": "#7cb7d4",
            "act": "#d4c47c", 
            "result": "#c47cd4",
        }.get(event_type, "#8a9a8a")
    
    def _sync_chain_to_events(self):
        """Sync chain selection to events selection."""
        if self.events_chrono and 0 <= self.chain_index < len(self.events_chrono):
            # Chain is chronological, events_list is reverse chronological
            self.selected_index = len(self.events_list) - 1 - self.chain_index
            self._update_events_selection()
    
    def _sync_events_to_chain(self):
        """Sync events selection to chain selection."""
        if self.events_list:
            self.chain_index = len(self.events_list) - 1 - self.selected_index
            self.refresh_chain()
    
    def _update_events_selection(self):
        """Update events list and related displays."""
        list_view = self.query_one("#event-list", ListView)
        if self.selected_index < len(list_view.children):
            list_view.index = self.selected_index
        self.refresh_detail()
        self.refresh_timeline()
        self.refresh_chain()
    
    # =========================================================================
    # FOCUS ACTIONS
    # =========================================================================
    
    def action_cycle_focus(self):
        """Cycle focus to next pane."""
        focus_order = [self.FOCUS_EVENTS, self.FOCUS_CHAIN, self.FOCUS_TIMELINE]
        idx = focus_order.index(self.current_focus)
        self.current_focus = focus_order[(idx + 1) % len(focus_order)]
        self._update_focus_visuals()
    
    def action_cycle_focus_back(self):
        """Cycle focus to previous pane."""
        focus_order = [self.FOCUS_EVENTS, self.FOCUS_CHAIN, self.FOCUS_TIMELINE]
        idx = focus_order.index(self.current_focus)
        self.current_focus = focus_order[(idx - 1) % len(focus_order)]
        self._update_focus_visuals()
    
    # =========================================================================
    # NAVIGATION ACTIONS
    # =========================================================================
    
    def action_move_up(self):
        """Move up in current focus."""
        if self.current_focus == self.FOCUS_EVENTS:
            if self.events_list and self.selected_index > 0:
                self.selected_index -= 1
                self._sync_events_to_chain()
                self._update_events_selection()
    
    def action_move_down(self):
        """Move down in current focus."""
        if self.current_focus == self.FOCUS_EVENTS:
            if self.events_list and self.selected_index < len(self.events_list) - 1:
                self.selected_index += 1
                self._sync_events_to_chain()
                self._update_events_selection()
    
    def action_move_left(self):
        """Move left - in chain or switch branch."""
        if self.current_focus == self.FOCUS_CHAIN:
            # Move back in chain (older)
            if self.chain_index > 0:
                self.chain_index -= 1
                self._sync_chain_to_events()
        elif self.current_focus == self.FOCUS_TIMELINE:
            # Switch branch
            self._switch_branch(-1)
        else:
            # In events, switch branch
            self._switch_branch(-1)
    
    def action_move_right(self):
        """Move right - in chain or switch branch."""
        if self.current_focus == self.FOCUS_CHAIN:
            # Move forward in chain (newer)
            if self.chain_index < len(self.events_chrono) - 1:
                self.chain_index += 1
                self._sync_chain_to_events()
        elif self.current_focus == self.FOCUS_TIMELINE:
            # Switch branch
            self._switch_branch(1)
        else:
            # In events, switch branch
            self._switch_branch(1)
    
    def _switch_branch(self, direction: int):
        """Switch to next/prev branch."""
        if len(self.branches_list) > 1:
            idx = self.branches_list.index(self.current_branch)
            self.current_branch = self.branches_list[(idx + direction) % len(self.branches_list)]
            self.selected_index = 0
            self.refresh_events()
            self.refresh_timeline()
            self.refresh_chain()
            self.update_status()
    
    # =========================================================================
    # CHAIN WALK (GLOBAL SHORTCUTS)
    # =========================================================================
    
    def action_chain_back(self):
        """Walk chain backward (to parent) - global shortcut."""
        if self.chain_index > 0:
            self.chain_index -= 1
            self._sync_chain_to_events()
    
    def action_chain_forward(self):
        """Walk chain forward (to child) - global shortcut."""
        if self.chain_index < len(self.events_chrono) - 1:
            self.chain_index += 1
            self._sync_chain_to_events()
    
    # =========================================================================
    # OTHER ACTIONS
    # =========================================================================
    
    def action_select(self):
        """Select/inspect current item."""
        if self.current_focus == self.FOCUS_CHAIN:
            # Jump to this event in events list
            self._sync_chain_to_events()
            self.current_focus = self.FOCUS_EVENTS
            self._update_focus_visuals()
        elif self.current_focus == self.FOCUS_EVENTS:
            # Open inspector
            if self.events_list and self.selected_index < len(self.events_list):
                event = self.events_list[self.selected_index]
                self.app.push_screen(InspectorScreen(self.substrate, event))
    
    def action_verify(self):
        """Verify hash chain integrity."""
        try:
            valid = self.substrate.verify(self.current_branch)
            if valid:
                self.notify(f"✓ Chain verified — '{self.current_branch}' intact", severity="information")
            else:
                self.notify(f"✗ Chain broken on '{self.current_branch}'!", severity="error")
        except Exception as e:
            self.notify(f"✗ Error: {e}", severity="error")
    
    def action_inspect(self):
        """Open full inspector for selected event."""
        if self.events_list and self.selected_index < len(self.events_list):
            event = self.events_list[self.selected_index]
            self.app.push_screen(InspectorScreen(self.substrate, event))
    
    def action_create_branch(self):
        """Create a branch from selected event."""
        if self.events_list and self.selected_index < len(self.events_list):
            event = self.events_list[self.selected_index]
            self.app.push_screen(BranchScreen(self.substrate, event, self._on_branch_created))
    
    def _on_branch_created(self, branch_name: str):
        """Callback when branch is created."""
        self.refresh_branches()
        self.current_branch = branch_name
        self.refresh_events()
        self.refresh_timeline()
        self.refresh_chain()
        self.update_status()
    
    def action_diff(self):
        """Open diff view."""
        if len(self.branches_list) > 1:
            self.app.push_screen(DiffScreen(self.substrate, self.branches_list))
        else:
            self.notify("Need at least 2 branches to diff", severity="warning")
    
    def action_show_guide(self):
        """Show the guide overlay."""
        self.app.push_screen(GuideOverlay(0))
    
    def action_back(self):
        """Go back."""
        self.app.pop_screen()
    
    def on_list_view_selected(self, event: ListView.Selected):
        """Handle list click selection."""
        if event.item and hasattr(event.item, 'event'):
            for i, e in enumerate(self.events_list):
                if e.id == event.item.event.id:
                    self.selected_index = i
                    break
            self._sync_events_to_chain()
            self._update_events_selection()


# =============================================================================
# INSPECTOR SCREEN
# =============================================================================

class InspectorScreen(Screen):
    """Full-screen event inspector."""
    
    BINDINGS = [
        Binding("q", "back", "Back"),
        Binding("escape", "back", "Back"),
        Binding("j", "jump_parent", "Jump to Parent"),
    ]
    
    def __init__(self, substrate: Substrate, event: Event):
        super().__init__()
        self.substrate = substrate
        self.event = event
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with ScrollableContainer(id="inspector"):
            # Event metadata
            yield Static(f"[bold]EVENT INSPECTOR[/bold]", id="detail-title")
            yield Static("")
            
            yield Static(f"[dim]ID:[/dim]        {self.event.id}")
            yield Static(f"[dim]Type:[/dim]      {self.event.type}")
            yield Static(f"[dim]Agent:[/dim]     {self.event.agent}")
            yield Static(f"[dim]Branch:[/dim]    {self.event.branch}")
            yield Static(f"[dim]Timestamp:[/dim] {format_ts_full(self.event.ts)}")
            
            yield Static("")
            yield Static("[bold]HASH CHAIN[/bold]")
            yield Static("")
            
            yield Static(f"[dim]Hash:[/dim]   [green bold]{self.event.hash}[/]")
            parent_display = self.event.parent or "(genesis - first event)"
            yield Static(f"[dim]Parent:[/dim] {parent_display}")
            
            if self.event.parent:
                yield Static("")
                yield Static("[dim]Press 'j' to jump to parent event[/dim]")
            
            yield Static("")
            yield Static("[bold]CONTENT[/bold]")
            yield Static("")
            
            content_log = RichLog(markup=True, wrap=True, id="content-box")
            yield content_log
        
        yield Footer()
    
    def on_mount(self):
        """Populate content."""
        content_log = self.query_one("#content-box", RichLog)
        content_str = json.dumps(self.event.content, indent=2)
        for line in content_str.split("\n"):
            content_log.write(line)
    
    def action_jump_parent(self):
        """Jump to parent event."""
        if self.event.parent:
            # Find parent event
            events = self.substrate.log(limit=1000, branch=self.event.branch)
            for e in events:
                if e.hash == self.event.parent:
                    self.app.pop_screen()
                    self.app.push_screen(InspectorScreen(self.substrate, e))
                    return
            self.notify("Parent event not found", severity="warning")
        else:
            self.notify("This is the genesis event (no parent)", severity="information")
    
    def action_back(self):
        self.app.pop_screen()


# =============================================================================
# BRANCH SCREEN
# =============================================================================

class BranchScreen(ModalScreen):
    """Create branch dialog."""
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]
    
    def __init__(self, substrate: Substrate, event: Event, callback=None):
        super().__init__()
        self.substrate = substrate
        self.event = event
        self.callback = callback
    
    def compose(self) -> ComposeResult:
        with Container(id="guide-overlay"):
            with Vertical(id="guide-box"):
                yield Static("[bold]CREATE BRANCH[/bold]", id="guide-title")
                yield Static(f"Fork from: {self.event.type} ({self.event.hash[:12]}...)")
                yield Static("")
                yield Static("Branch name:")
                yield Input(placeholder="fix", id="branch-name")
                yield Static("")
                yield Button("Create", id="create-btn", variant="primary")
    
    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "create-btn":
            name_input = self.query_one("#branch-name", Input)
            name = name_input.value.strip()
            if name:
                try:
                    self.substrate.branch(name, from_event=self.event.id)
                    self.app.pop_screen()
                    if self.callback:
                        self.callback(name)
                except Exception as e:
                    self.notify(f"Error: {e}", severity="error")
    
    def action_cancel(self):
        self.app.pop_screen()


# =============================================================================
# DIFF SCREEN - FIXED
# =============================================================================

class DiffScreen(Screen):
    """Side-by-side branch comparison."""
    
    BINDINGS = [
        Binding("q", "back", "Back"),
        Binding("escape", "back", "Back"),
    ]
    
    def __init__(self, substrate: Substrate, branches: List[str]):
        super().__init__()
        self.substrate = substrate
        self.branches = branches
        self.left_branch = branches[0] if branches else "main"
        self.right_branch = branches[1] if len(branches) > 1 else "main"
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        # Fixed horizontal layout
        with Horizontal(id="diff-container"):
            # Left branch
            with Vertical(id="diff-left"):
                yield Static(f"[bold green]⎇ {self.left_branch}[/bold green]", id="diff-left-title", classes="diff-title")
                yield RichLog(id="diff-left-content", markup=True, wrap=True)
            
            # Right branch  
            with Vertical(id="diff-right"):
                yield Static(f"[bold yellow]⎇ {self.right_branch}[/bold yellow]", id="diff-right-title", classes="diff-title")
                yield RichLog(id="diff-right-content", markup=True, wrap=True)
        
        yield Static("[q] Back  │  Comparing branches side-by-side", id="status-bar")
        yield Footer()
    
    def on_mount(self):
        """Load diff content."""
        self._refresh_diff()
    
    def _refresh_diff(self):
        """Refresh both sides."""
        left_log = self.query_one("#diff-left-content", RichLog)
        right_log = self.query_one("#diff-right-content", RichLog)
        
        left_log.clear()
        right_log.clear()
        
        left_events = self.substrate.log(limit=50, branch=self.left_branch)
        right_events = self.substrate.log(limit=50, branch=self.right_branch)
        
        # Left side
        left_log.write(f"[bold]{len(left_events)} events[/bold]")
        left_log.write("")
        for e in left_events:
            left_log.write(f"[#6bbd8a]{e.type:8}[/] [dim]{e.hash[:8]}[/dim]")
            # Content preview with wrapping
            content_preview = str(e.content)
            if len(content_preview) > 60:
                content_preview = content_preview[:60] + "..."
            left_log.write(f"  {content_preview}")
            left_log.write("")
        
        # Right side
        right_log.write(f"[bold]{len(right_events)} events[/bold]")
        right_log.write("")
        for e in right_events:
            right_log.write(f"[#c4956a]{e.type:8}[/] [dim]{e.hash[:8]}[/dim]")
            content_preview = str(e.content)
            if len(content_preview) > 60:
                content_preview = content_preview[:60] + "..."
            right_log.write(f"  {content_preview}")
            right_log.write("")
    
    def action_back(self):
        self.app.pop_screen()


# =============================================================================
# MAIN APP
# =============================================================================

class StudioApp(App):
    """krnx Studio application."""
    
    CSS = CSS
    TITLE = "krnx Studio"
    
    def __init__(self, initial_workspace: Optional[str] = None, guided: bool = False, path: Optional[str] = None):
        super().__init__()
        self.initial_workspace = initial_workspace
        self.guided = guided
        self.path = path
    
    def on_mount(self):
        """Start the app."""
        if self.initial_workspace:
            self.open_workspace(self.initial_workspace)
        else:
            self.push_screen(MainMenuScreen())
    
    def open_workspace(self, name: str):
        """Open a workspace."""
        self.push_screen(TimelineScreen(name, guided=self.guided, path=self.path))


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_studio(workspace: Optional[str] = None, guided: bool = False, path: Optional[str] = None):
    """Run krnx Studio."""
    app = StudioApp(initial_workspace=workspace, guided=guided, path=path)
    app.run()


if __name__ == "__main__":
    run_studio()
