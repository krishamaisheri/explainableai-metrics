"""
Pretty-print module for structured CLI output.

Uses ANSI escape codes for colored, well-formatted pipeline results.
Works on Windows via colorama.
"""

import sys
import textwrap
import datetime

# ── ANSI color codes (Windows-safe via colorama init) ───────────
try:
    import colorama
    colorama.init(autoreset=True)
except ImportError:
    pass  # colors will still work on most modern terminals

# Colors
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
ITALIC  = "\033[3m"

# Foreground
WHITE   = "\033[97m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
GRAY    = "\033[90m"

# Background
BG_DARK     = "\033[48;5;235m"
BG_HEADER   = "\033[48;5;24m"
BG_GREEN    = "\033[48;5;22m"
BG_RED      = "\033[48;5;52m"
BG_YELLOW   = "\033[48;5;58m"

# Symbols
BAR_FILLED = "█"
BAR_EMPTY  = "░"
CHECK      = "✓"
CROSS      = "✗"
ARROW      = "►"
DOT        = "●"

WIDTH = 90


def _hr(char="─", color=GRAY):
    print(f"{color}{char * WIDTH}{RESET}")


def _centered(text, color=WHITE):
    padding = (WIDTH - len(text)) // 2
    print(f"{' ' * padding}{color}{text}{RESET}")


def _bar(score: float, width: int = 30) -> str:
    """Generate a colored progress bar."""
    filled = int(score * width)
    empty = width - filled
    if score >= 0.9:
        color = GREEN
    elif score >= 0.7:
        color = YELLOW
    else:
        color = RED
    return f"{color}{BAR_FILLED * filled}{GRAY}{BAR_EMPTY * empty}{RESET}"


def _severity_color(severity: str) -> str:
    return {
        "Red": RED,
        "Amber": YELLOW,
        "Green": GREEN,
    }.get(severity, WHITE)


def _severity_icon(severity: str) -> str:
    return {
        "Red": f"{RED}🔴",
        "Amber": f"{YELLOW}🟡",
        "Green": f"{GREEN}🟢",
    }.get(severity, "⚪")


def print_header(query: str):
    """Print the pipeline header."""
    print()
    _hr("═", CYAN)
    _centered("EXPLAINABILITY METRICS PIPELINE", BOLD + CYAN)
    _hr("═", CYAN)
    print()
    print(f"  {BOLD}{CYAN}Query:{RESET}  {WHITE}{query}{RESET}")
    print(f"  {GRAY}Time:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    _hr()
    print()


def print_explanation(explanation: str):
    """Print the generated explanation in a bordered box."""
    print(f"  {BOLD}{MAGENTA}{ARROW} LLM RESPONSE{RESET}")
    _hr("─", MAGENTA)

    wrapped = textwrap.fill(explanation, width=WIDTH - 6)
    for line in wrapped.split("\n"):
        print(f"  {DIM}{WHITE}{line}{RESET}")

    _hr("─", MAGENTA)
    print()


def print_metrics(result: dict):
    """Print the metrics table with scores, timing, and LLM call info."""
    scores = result.get("metric_scores", {})
    timings = result.get("metric_timings", {})
    failures = result.get("metric_failures", {})
    llm_stats = result.get("llm_stats", {})

    print(f"  {BOLD}{BLUE}{ARROW} METRIC SCORES{RESET}")
    _hr("─", BLUE)

    # Table header
    print(f"  {BOLD}{WHITE}{'Metric':<8} {'Score':>7}  {'Bar':<32} {'Time':>7}  {'LLM Calls':>10}  {'Model':<30}  {'Status':<6}{RESET}")
    print(f"  {GRAY}{'─'*8} {'─'*7}  {'─'*30}  {'─'*7}  {'─'*10}  {'─'*30}  {'─'*6}{RESET}")

    metric_order = ["IACS", "ICR", "IRCS", "EDAS", "SECS", "PGSS", "ESI", "EDR"]

    for name in metric_order:
        if name not in scores:
            continue

        score = scores[name]
        time_s = timings.get(name, 0.0)
        bar = _bar(score)
        stat = llm_stats.get(name, {})
        calls = stat.get("total_calls", 0)
        models = ", ".join(stat.get("models_used", ["-"]))
        failed = stat.get("failed", 0)

        if name in failures:
            status = f"{RED}{CROSS} FAIL{RESET}"
        elif failed > 0:
            status = f"{YELLOW}⚠ {failed}err{RESET}"
        else:
            status = f"{GREEN}{CHECK} OK{RESET}"

        # Truncate model name for display
        model_display = models if len(models) <= 30 else models[:27] + "..."

        print(
            f"  {CYAN}{name:<8}{RESET} "
            f"{WHITE}{score:>7.4f}{RESET}  "
            f"{bar}  "
            f"{GRAY}{time_s:>6.1f}s{RESET}  "
            f"{WHITE}{calls:>10}{RESET}  "
            f"{DIM}{model_display:<30}{RESET}  "
            f"{status}"
        )

    _hr("─", BLUE)
    print()


def print_aggregate(result: dict):
    """Print the aggregate score with a large visual indicator."""
    agg = result.get("aggregate_score", 0.0)

    if agg >= 0.9:
        color = GREEN
        label = "EXCELLENT"
    elif agg >= 0.7:
        color = YELLOW
        label = "ACCEPTABLE"
    elif agg >= 0.5:
        color = YELLOW
        label = "NEEDS IMPROVEMENT"
    else:
        color = RED
        label = "POOR"

    print(f"  {BOLD}{color}{ARROW} AGGREGATE SCORE{RESET}")
    _hr("─", color)
    bar = _bar(agg, width=50)
    print(f"  {BOLD}{color}{agg:.4f}{RESET}  {bar}  {BOLD}{color}{label}{RESET}")
    _hr("─", color)
    print()


def print_alerts(result: dict):
    """Print alerts color-coded by severity."""
    alerts = result.get("alerts", [])

    if not alerts:
        print(f"  {BOLD}{GREEN}{ARROW} ALERTS: None — all metrics within thresholds{RESET}")
        print()
        return

    print(f"  {BOLD}{YELLOW}{ARROW} ALERTS ({len(alerts)} triggered){RESET}")
    _hr("─", YELLOW)

    for alert in alerts:
        icon = _severity_icon(alert["severity"])
        color = _severity_color(alert["severity"])
        print(
            f"  {icon} {color}{BOLD}{alert['metric']:<12}{RESET}  "
            f"Score: {color}{alert['score']:.4f}{RESET}  "
            f"{GRAY}(green ≥ {alert['threshold_green']}, amber ≥ {alert['threshold_amber']}){RESET}"
        )

    _hr("─", YELLOW)
    print()


def print_llm_summary(result: dict):
    """Print LLM call summary statistics."""
    llm_global = result.get("llm_global", {})
    llm_stats = result.get("llm_stats", {})
    failures = result.get("metric_failures", {})

    print(f"  {BOLD}{GRAY}{ARROW} LLM CALL SUMMARY{RESET}")
    _hr("─", GRAY)

    total = llm_global.get("total_calls", 0)
    success = llm_global.get("successful", 0)
    failed = llm_global.get("failed", 0)
    total_time = llm_global.get("total_time", 0.0)

    print(f"  {WHITE}Total LLM Calls:    {BOLD}{CYAN}{total}{RESET}")
    print(f"  {WHITE}Successful:         {GREEN}{success}{RESET}")
    print(f"  {WHITE}Failed:             {RED if failed > 0 else GREEN}{failed}{RESET}")
    print(f"  {WHITE}Total LLM Time:     {CYAN}{total_time:.1f}s{RESET}")

    if failures:
        print()
        print(f"  {RED}{BOLD}Metric Computation Failures:{RESET}")
        for metric, err in failures.items():
            print(f"    {RED}{CROSS} {metric}: {DIM}{err[:80]}{RESET}")

    # Per-metric breakdown
    if llm_stats:
        print()
        print(f"  {BOLD}{WHITE}Per-Metric Breakdown:{RESET}")
        print(f"  {GRAY}{'Metric':<8} {'Calls':>6} {'Success':>8} {'Failed':>7} {'Time':>8}  {'Model(s)'}{RESET}")
        print(f"  {GRAY}{'─'*8} {'─'*6} {'─'*8} {'─'*7} {'─'*8}  {'─'*25}{RESET}")

        for metric in sorted(llm_stats.keys()):
            s = llm_stats[metric]
            models = ", ".join(s.get("models_used", []))
            failed_c = s.get("failed", 0)
            failed_color = RED if failed_c > 0 else GREEN
            print(
                f"  {CYAN}{metric:<8}{RESET} "
                f"{WHITE}{s['total_calls']:>6}{RESET} "
                f"{GREEN}{s['successful']:>8}{RESET} "
                f"{failed_color}{failed_c:>7}{RESET} "
                f"{GRAY}{s['total_time']:>7.1f}s{RESET}  "
                f"{DIM}{models}{RESET}"
            )

    _hr("─", GRAY)
    print()


def print_result(result: dict, query: str = ""):
    """Print the full structured pipeline result."""
    q = query or result.get("query", "")
    print_header(q)
    print_explanation(result.get("explanation", ""))
    print_metrics(result)
    print_aggregate(result)
    print_alerts(result)
    print_llm_summary(result)
    _hr("═", CYAN)
    print()


def print_batch_summary(results: list[dict]):
    """Print a summary table after a batch run."""
    print()
    _hr("═", MAGENTA)
    _centered("BATCH RUN SUMMARY", BOLD + MAGENTA)
    _hr("═", MAGENTA)
    print()

    print(f"  {BOLD}{WHITE}{'#':>3}  {'Query':<55} {'Agg Score':>10} {'LLM Calls':>10} {'Time':>8} {'Failures':>9}{RESET}")
    print(f"  {GRAY}{'─'*3}  {'─'*55} {'─'*10} {'─'*10} {'─'*8} {'─'*9}{RESET}")

    total_time = 0.0
    total_calls = 0
    total_failures = 0

    for i, r in enumerate(results, 1):
        q = r.get("query", "")[:53]
        agg = r.get("aggregate_score", 0.0)
        llm_g = r.get("llm_global", {})
        calls = llm_g.get("total_calls", 0)
        t = sum(r.get("metric_timings", {}).values())
        fails = len(r.get("metric_failures", {}))

        total_time += t
        total_calls += calls
        total_failures += fails

        if agg >= 0.9:
            agg_color = GREEN
        elif agg >= 0.7:
            agg_color = YELLOW
        else:
            agg_color = RED

        fail_color = RED if fails > 0 else GREEN

        print(
            f"  {GRAY}{i:>3}{RESET}  "
            f"{WHITE}{q:<55}{RESET} "
            f"{agg_color}{agg:>10.4f}{RESET} "
            f"{CYAN}{calls:>10}{RESET} "
            f"{GRAY}{t:>7.1f}s{RESET} "
            f"{fail_color}{fails:>9}{RESET}"
        )

    print(f"  {GRAY}{'─'*3}  {'─'*55} {'─'*10} {'─'*10} {'─'*8} {'─'*9}{RESET}")
    print(
        f"  {BOLD}{WHITE}{'':>3}  {'TOTAL':<55} {'':>10} "
        f"{CYAN}{total_calls:>10}{RESET} "
        f"{GRAY}{total_time:>7.1f}s{RESET} "
        f"{RED if total_failures > 0 else GREEN}{total_failures:>9}{RESET}"
    )

    _hr("═", MAGENTA)
    print()
