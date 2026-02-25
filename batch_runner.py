"""
Batch Query Runner.

Runs a predefined set of test queries through the pipeline
and prints structured results + a summary table.

Usage:
    python batch_runner.py
"""

import logging
import time
import pretty_print
from pipeline import evaluate

# Suppress noisy logs during batch run
logging.basicConfig(level=logging.WARNING)

QUERIES = [

    # Q1 — Universal Credit: low income + children
    "I am unemployed, have two children, and earn no income. Can I claim Universal Credit?",

    # Q2 — Universal Credit: higher income
    "I work full-time and earn £3,200 per month. Am I eligible for Universal Credit?",

    # Q3 — Housing Benefit appeal
    "My Housing Benefit application was rejected. How can I appeal the decision?",

    # Q4 — Housing Benefit eligibility
    "I rent privately and receive Pension Credit. Can I apply for Housing Benefit?",

    # Q5 — Disability support (general)
    "I have a long-term disability that affects my ability to work. What financial support can I claim?",

    # Q6 — DLA child case
    "My 10-year-old child has mobility and care needs. Can they receive Disability Living Allowance?",

    # Q7 — DLA adult confusion
    "I am 30 years old with a physical disability. Can I apply for DLA?",

    # Q8 — Universal Credit + fluctuating income
    "I am self-employed and my income changes between £400 and £1,200 per month. How does this affect my Universal Credit?",

    # Q9 — Housing Benefit + employment
    "I work part-time and rent from a private landlord. Can I still get Housing Benefit?",

    # Q10 — Multi-policy overlap case
    "I am unemployed, disabled, and rent privately. Should I apply for Universal Credit, Housing Benefit, or DLA?"

]


def main():
    print()
    print(f"{pretty_print.BOLD}{pretty_print.CYAN}")
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║        EXPLAINABILITY METRICS — BATCH EVALUATION            ║")
    print(f"  ║        Running {len(QUERIES)} queries sequentially...                  ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print(f"{pretty_print.RESET}")

    results = []
    batch_start = time.time()

    for i, query in enumerate(QUERIES, 1):
        print(f"\n{pretty_print.BOLD}{pretty_print.MAGENTA}{'═' * 90}{pretty_print.RESET}")
        print(f"  {pretty_print.BOLD}{pretty_print.WHITE}QUERY {i}/{len(QUERIES)}{pretty_print.RESET}")
        print(f"{pretty_print.BOLD}{pretty_print.MAGENTA}{'═' * 90}{pretty_print.RESET}\n")

        try:
            result = evaluate(query, silent=False)
            results.append(result)
        except Exception as exc:
            print(f"  {pretty_print.RED}FATAL ERROR on Q{i}: {exc}{pretty_print.RESET}")
            results.append({
                "query": query,
                "aggregate_score": 0.0,
                "metric_timings": {},
                "metric_failures": {"PIPELINE": str(exc)},
                "llm_global": {"total_calls": 0},
            })

    batch_time = round(time.time() - batch_start, 1)

    # Print batch summary
    pretty_print.print_batch_summary(results)

    print(f"  {pretty_print.BOLD}{pretty_print.CYAN}Total batch time: {batch_time}s{pretty_print.RESET}")
    print()


if __name__ == "__main__":
    main()
