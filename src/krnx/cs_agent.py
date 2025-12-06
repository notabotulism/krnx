"""
krnx Customer Service Agent Demo

A simple refund-approval agent that demonstrates krnx instrumentation.
Used for:
  - `krnx try` — BYOK live demo
  - Video capture for marketing

The scenario:
  1. Customer requests $8,000 refund
  2. Agent checks order history (looks legitimate)
  3. Agent approves → BAD OUTCOME (this was fraud)
  4. Branch from step 1, add fraud-check context
  5. Agent denies → GOOD OUTCOME
  6. Both timelines preserved, verifiable

Usage:
    export OPENAI_API_KEY=sk-...
    # or
    export ANTHROPIC_API_KEY=sk-...
    krnx try
"""

import os
import json
import time
from typing import Optional, Tuple
from dataclasses import dataclass


def get_api_key() -> Tuple[Optional[str], Optional[str]]:
    """Get API key from environment. Returns (key, provider)."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if anthropic_key:
        return anthropic_key, "anthropic"
    if openai_key:
        return openai_key, "openai"
    return None, None


def check_provider_available(provider: str) -> bool:
    """Check if provider package is installed."""
    try:
        if provider == "anthropic":
            import anthropic
        elif provider == "openai":
            import openai
        return True
    except ImportError:
        return False


@dataclass
class RefundRequest:
    """A customer refund request."""
    customer_id: str
    customer_name: str
    order_id: str
    amount: float
    reason: str
    order_history: list
    fraud_score: Optional[float] = None  # 0-100, higher = more suspicious
    customer_tier: Optional[str] = None


# Demo scenarios
SCENARIO_MAIN = RefundRequest(
    customer_id="cust_8x7k2m",
    customer_name="Alex Chen",
    order_id="ord_j3k8n2",
    amount=8000.00,
    reason="Item not as described, requesting full refund",
    order_history=[
        {"order_id": "ord_a1b2c3", "amount": 450.00, "status": "completed", "date": "2024-08"},
        {"order_id": "ord_d4e5f6", "amount": 1200.00, "status": "completed", "date": "2024-10"},
        {"order_id": "ord_j3k8n2", "amount": 8000.00, "status": "delivered", "date": "2024-11"},
    ],
    # No fraud score in main scenario — agent doesn't check
)

SCENARIO_FIX = RefundRequest(
    customer_id="cust_8x7k2m",
    customer_name="Alex Chen",
    order_id="ord_j3k8n2",
    amount=8000.00,
    reason="Item not as described, requesting full refund",
    order_history=[
        {"order_id": "ord_a1b2c3", "amount": 450.00, "status": "completed", "date": "2024-08"},
        {"order_id": "ord_d4e5f6", "amount": 1200.00, "status": "completed", "date": "2024-10"},
        {"order_id": "ord_j3k8n2", "amount": 8000.00, "status": "delivered", "date": "2024-11"},
    ],
    fraud_score=87.3,  # High fraud score — agent should deny
    customer_tier="standard",  # Not a VIP
)


SYSTEM_PROMPT = """You are a customer service agent handling refund requests.

Your job is to decide whether to APPROVE or DENY refund requests based on the information provided.

Guidelines:
- Check order history for patterns
- Consider the refund amount relative to customer history
- If fraud_score is provided and > 70, this is a HIGH RISK request — be very careful
- If fraud_score is not provided, you don't have that data available
- Large refunds (>$5000) on accounts with limited history are risky

You MUST respond with a JSON object containing:
{
  "decision": "APPROVE" or "DENY",
  "reasoning": "brief explanation",
  "confidence": 0.0 to 1.0
}

Respond ONLY with the JSON object, no other text."""


# Mock responses for scripted demo (no API key needed)
MOCK_RESPONSES = {
    "main": {
        "decision": "APPROVE",
        "reasoning": "Customer has consistent order history with completed purchases. The refund amount aligns with their most recent order. No fraud indicators available.",
        "confidence": 0.78,
    },
    "fix": {
        "decision": "DENY",
        "reasoning": "HIGH RISK: Fraud score of 87.3 exceeds threshold. Despite order history, this request shows strong indicators of account compromise or fraudulent activity.",
        "confidence": 0.92,
    },
}


def run_agent(substrate, request: RefundRequest, branch: str = "main", mock: bool = False, provider: str = "openai") -> dict:
    """
    Run the CS agent on a refund request.
    
    Records all events to krnx substrate:
      - observe: incoming request
      - think: LLM reasoning
      - act: decision
      - result: outcome
      
    Args:
        substrate: krnx Substrate instance
        request: RefundRequest to process
        branch: Branch to record events on
        mock: If True, use mock responses instead of real LLM calls
        provider: "anthropic" or "openai"
    """
    model_name = None
    
    if not mock:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            model_name = "claude-sonnet-4-20250514"
        else:  # openai
            import openai
            client = openai.OpenAI()
            model_name = "gpt-4o"
    
    # === OBSERVE ===
    observe_content = {
        "customer_id": request.customer_id,
        "customer_name": request.customer_name,
        "order_id": request.order_id,
        "amount": request.amount,
        "reason": request.reason,
        "order_history": request.order_history,
    }
    if request.fraud_score is not None:
        observe_content["fraud_score"] = request.fraud_score
    if request.customer_tier is not None:
        observe_content["customer_tier"] = request.customer_tier
    
    substrate.record(
        "observe",
        observe_content,
        agent="cs-agent",
        branch=branch,
    )
    
    # === THINK ===
    # Build the prompt (needed for both mock and real)
    user_prompt = f"""Process this refund request:

Customer: {request.customer_name} ({request.customer_id})
Order: {request.order_id}
Amount: ${request.amount:,.2f}
Reason: {request.reason}

Order History:
{json.dumps(request.order_history, indent=2)}
"""
    if request.fraud_score is not None:
        user_prompt += f"\n⚠️ FRAUD SCORE: {request.fraud_score}/100 (HIGH RISK)\n"
    if request.customer_tier is not None:
        user_prompt += f"Customer Tier: {request.customer_tier}\n"
    
    if mock:
        # Use mock response
        decision_data = MOCK_RESPONSES[branch]
        raw_response = json.dumps(decision_data, indent=2)
        tokens_used = 150  # Fake token count
        model_name = "mock"
    else:
        # Call LLM based on provider
        if provider == "anthropic":
            response = client.messages.create(
                model=model_name,
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            raw_response = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
        else:  # openai
            response = client.chat.completions.create(
                model=model_name,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            )
            raw_response = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
        
        # Parse response
        try:
            decision_data = json.loads(raw_response)
        except json.JSONDecodeError:
            decision_data = {
                "decision": "ERROR",
                "reasoning": f"Failed to parse: {raw_response[:100]}",
                "confidence": 0.0
            }
    
    # Record the actual prompts and responses for full auditability
    think_content = {
        "model": model_name,
        "tokens_used": tokens_used,
        # Full prompts — the whole point of krnx
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        # Raw LLM response
        "raw_response": raw_response,
        # Parsed result
        "parsed": decision_data,
    }
    
    substrate.record(
        "think",
        think_content,
        agent="cs-agent",
        branch=branch,
    )
    
    # === ACT ===
    decision = decision_data.get("decision", "ERROR")
    
    substrate.record(
        "act",
        {
            "action": decision,
            "order_id": request.order_id,
            "amount": request.amount,
            "customer_id": request.customer_id,
        },
        agent="cs-agent",
        branch=branch,
    )
    
    # === RESULT ===
    # Simulate outcome — this is where we reveal it was fraud
    if decision == "APPROVE":
        # In the "real world" this would have been fraud
        is_fraud = request.fraud_score is None or request.fraud_score > 70
        if is_fraud and request.fraud_score is None:
            # Main scenario: approved without checking, was fraud
            result = {
                "outcome": "LOSS",
                "amount_lost": request.amount,
                "reason": "Approved without fraud check — account was compromised",
                "chargeback": True,
            }
        elif is_fraud:
            # Shouldn't happen if agent follows rules, but handle it
            result = {
                "outcome": "LOSS", 
                "amount_lost": request.amount,
                "reason": "Approved despite high fraud score",
                "chargeback": True,
            }
        else:
            result = {
                "outcome": "SUCCESS",
                "amount_refunded": request.amount,
                "customer_retained": True,
            }
    else:
        # DENY
        if request.fraud_score and request.fraud_score > 70:
            result = {
                "outcome": "FRAUD_PREVENTED",
                "amount_saved": request.amount,
                "reason": "Correctly identified high-risk request",
            }
        else:
            result = {
                "outcome": "DENIED",
                "reason": decision_data.get("reasoning", "Request denied"),
            }
    
    substrate.record(
        "result",
        result,
        agent="cs-agent",
        branch=branch,
    )
    
    return {
        "decision": decision,
        "result": result,
        "events_recorded": 4,
    }


def run_demo(substrate, verbose: bool = True, mock: bool = False, provider: str = "openai") -> dict:
    """
    Run the full demo scenario:
    1. Main branch — agent approves (mistake)
    2. Create 'fix' branch from first event
    3. Fix branch — agent denies (correct)
    4. Return both outcomes
    
    Args:
        substrate: krnx Substrate instance
        verbose: Print progress to console
        mock: Use mock LLM responses (no API key needed)
        provider: "anthropic" or "openai"
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    
    console = Console()
    
    if verbose:
        console.print(Panel.fit(
            "[bold]krnx Customer Service Agent Demo[/bold]\n\n"
            "This demo shows:\n"
            "• Real LLM calls (Claude) making decisions\n"
            "• Every step recorded to krnx timeline\n"
            "• Hash chain building in real-time\n"
            "• Branching to explore alternate outcomes\n"
            "• Verification of both timelines",
            border_style="green"
        ))
        console.print()
    
    # === MAIN BRANCH ===
    if verbose:
        console.print("[bold cyan]━━━ MAIN TIMELINE ━━━[/bold cyan]")
        console.print(f"Processing refund request: [yellow]${SCENARIO_MAIN.amount:,.2f}[/yellow] for {SCENARIO_MAIN.customer_name}")
        console.print("[dim]No fraud check available...[/dim]")
        console.print()
    
    main_result = run_agent(substrate, SCENARIO_MAIN, branch="main", mock=mock, provider=provider)
    
    if verbose:
        if main_result["result"]["outcome"] == "LOSS":
            console.print(f"[bold red]✗ DECISION: {main_result['decision']}[/bold red]")
            console.print(f"[red]  → LOSS: ${main_result['result']['amount_lost']:,.2f}[/red]")
            console.print(f"[dim]  {main_result['result']['reason']}[/dim]")
        else:
            console.print(f"[green]✓ DECISION: {main_result['decision']}[/green]")
        console.print()
    
    # Get first event for branching
    events = substrate.log(branch="main", limit=1)
    if not events:
        raise RuntimeError("No events recorded")
    first_event = events[0]
    
    # === CREATE BRANCH ===
    if verbose:
        console.print(f"[bold magenta]━━━ BRANCHING ━━━[/bold magenta]")
        console.print(f"Creating branch 'fix' from event [cyan]{first_event.id}[/cyan]")
        console.print(f"[dim]Hash: {first_event.hash}[/dim]")
        console.print()
    
    substrate.branch("fix", from_event=first_event.id)
    
    # === FIX BRANCH ===
    if verbose:
        console.print("[bold cyan]━━━ FIX TIMELINE ━━━[/bold cyan]")
        console.print(f"Processing same request WITH fraud check...")
        console.print(f"[yellow]⚠ Fraud score: {SCENARIO_FIX.fraud_score}/100[/yellow]")
        console.print()
    
    fix_result = run_agent(substrate, SCENARIO_FIX, branch="fix", mock=mock, provider=provider)
    
    if verbose:
        if fix_result["result"]["outcome"] == "FRAUD_PREVENTED":
            console.print(f"[bold green]✓ DECISION: {fix_result['decision']}[/bold green]")
            console.print(f"[green]  → SAVED: ${fix_result['result']['amount_saved']:,.2f}[/green]")
        else:
            console.print(f"[yellow]DECISION: {fix_result['decision']}[/yellow]")
        console.print()
    
    # === VERIFY BOTH ===
    if verbose:
        console.print("[bold cyan]━━━ VERIFICATION ━━━[/bold cyan]")
    
    main_valid = substrate.verify(branch="main")
    fix_valid = substrate.verify(branch="fix")
    
    if verbose:
        console.print(f"main branch: [green]✓ VALID[/green]" if main_valid else f"main branch: [red]✗ INVALID[/red]")
        console.print(f"fix branch:  [green]✓ VALID[/green]" if fix_valid else f"fix branch:  [red]✗ INVALID[/red]")
        console.print()
    
    # === SUMMARY ===
    if verbose:
        table = Table(title="Timeline Comparison", show_header=True)
        table.add_column("Branch", style="cyan")
        table.add_column("Decision", style="white")
        table.add_column("Outcome", style="white")
        table.add_column("Events", style="white")
        table.add_column("Verified", style="white")
        
        main_events = substrate.log(branch="main")
        fix_events = substrate.log(branch="fix")
        
        table.add_row(
            "main",
            main_result["decision"],
            f"[red]LOSS ${SCENARIO_MAIN.amount:,.2f}[/red]" if main_result["result"]["outcome"] == "LOSS" else main_result["result"]["outcome"],
            str(len(main_events)),
            "[green]✓[/green]" if main_valid else "[red]✗[/red]"
        )
        table.add_row(
            "fix",
            fix_result["decision"],
            f"[green]SAVED ${SCENARIO_FIX.amount:,.2f}[/green]" if fix_result["result"]["outcome"] == "FRAUD_PREVENTED" else fix_result["result"]["outcome"],
            str(len(fix_events)),
            "[green]✓[/green]" if fix_valid else "[red]✗[/red]"
        )
        
        console.print(table)
        console.print()
        
        # Show hash chain
        console.print("[bold]Hash Chain (main):[/bold]")
        for e in main_events[:4]:
            parent = e.parent[:8] if e.parent else "genesis"
            console.print(f"  {e.type:8} │ {parent} → [cyan]{e.hash[:8]}[/cyan]")
        console.print()
    
    return {
        "main": main_result,
        "fix": fix_result,
        "main_valid": main_valid,
        "fix_valid": fix_valid,
    }


def run_try_demo():
    """
    Entry point for `krnx try` command.
    
    Checks for API key (OpenAI or Anthropic), runs demo, offers to open Studio.
    """
    from rich.console import Console
    from rich.prompt import Confirm
    
    console = Console()
    
    # Check for API key (either provider)
    api_key, provider = get_api_key()
    if not api_key:
        console.print("[red]Error: No API key found[/red]")
        console.print("\nSet one of these:")
        console.print("  [cyan]export OPENAI_API_KEY=sk-...[/cyan]")
        console.print("  [cyan]export ANTHROPIC_API_KEY=sk-...[/cyan]")
        return 1
    
    # Check for provider package
    if not check_provider_available(provider):
        console.print(f"[red]Error: {provider} package not installed[/red]")
        console.print(f"Install with: [cyan]pip install {provider}[/cyan]")
        return 1
    
    console.print(f"[dim]Using {provider} API...[/dim]")
    
    # Import krnx
    from . import init as krnx_init
    
    # Create workspace
    console.print("[dim]Initializing krnx workspace 'try-demo'...[/dim]")
    substrate = krnx_init("try-demo")
    
    # Run demo
    try:
        results = run_demo(substrate, verbose=True, mock=False, provider=provider)
    except Exception as e:
        console.print(f"[red]Error running demo: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1
    
    # Offer to open Studio
    console.print()
    if Confirm.ask("Open krnx Studio to explore the timelines?", default=True):
        from .studio import run_studio
        run_studio("try-demo")
    
    return 0


def run_mock_demo():
    """
    Entry point for scripted demo (no API key needed).
    
    Uses mock LLM responses for video capture and quick demos.
    """
    from rich.console import Console
    from rich.prompt import Confirm
    import tempfile
    
    console = Console()
    
    # Import krnx
    from . import init as krnx_init
    
    # Create workspace in temp dir
    temp_dir = tempfile.mkdtemp(prefix="krnx_mock_demo_")
    console.print("[dim]Initializing krnx workspace (mock mode)...[/dim]")
    substrate = krnx_init("mock-demo", path=temp_dir)
    
    # Run demo with mock responses
    try:
        results = run_demo(substrate, verbose=True, mock=True)
    except Exception as e:
        console.print(f"[red]Error running demo: {e}[/red]")
        return 1
    
    # Offer to open Studio
    console.print()
    if Confirm.ask("Open krnx Studio to explore the timelines?", default=True):
        from .studio import run_studio
        run_studio("mock-demo", path=temp_dir)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_try_demo())
