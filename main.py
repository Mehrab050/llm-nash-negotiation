"""
Multi-Agent LLM Negotiation System for Construction Site
Robot Resource Allocation Using Nash Bargaining Game Theory

Author: Md Mehrab Hossain
Date: March 2026
"""

import os
import sys
import time
import logging
from typing import Dict

import config
from environment import ConstructionSiteSimulation, ScenarioMetrics
from llm_negotiator import LLMNegotiator
from visualizer import generate_all_figures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print project banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║   Multi-Agent LLM Negotiation for Construction Robotics        ║
║   Nash Bargaining Game Theory + LLM Explainability             ║        
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_scenario_results(name: str, metrics: ScenarioMetrics):
    """Print formatted results for a single scenario."""
    hours = metrics.total_completion_time / 60
    print(f"\n  {'─' * 50}")
    print(f"  {config.SCENARIO_LABELS[name]}")
    print(f"  {'─' * 50}")
    print(f"  Total Completion:       {metrics.total_completion_time:.0f} min "
          f"({hours:.1f} hrs)")
    print(f"  Conflicts Detected:     {metrics.num_conflicts}")
    print(f"  Avg Resolution Time:    {metrics.avg_conflict_resolution_ms:.1f} ms")
    print(f"  Disruption Recovery:    {metrics.disruption_recovery_time:.0f} min")
    print(f"  Schedule Stability:     {metrics.schedule_stability:.1%}")

    print(f"  Resource Utilization:")
    for res, util in metrics.resource_utilization.items():
        if res != "steel_materials":
            print(f"    {res:15s}: {util:.1f}%")


def print_comparison(all_metrics: Dict[str, ScenarioMetrics]):
    """Print side-by-side comparison of all scenarios."""
    print("\n" + "=" * 60)
    print("  SCENARIO COMPARISON")
    print("=" * 60)

    baseline = all_metrics["baseline"].total_completion_time
    for name in config.SCENARIOS:
        m = all_metrics[name]
        improvement = ((baseline - m.total_completion_time) / baseline) * 100
        sign = "↓" if improvement > 0 else "↑"
        print(
            f"  {config.SCENARIO_LABELS[name]:30s} "
            f"{m.total_completion_time:6.0f} min  "
            f"({sign} {abs(improvement):.1f}% vs baseline)"
        )

    print()
    print("  Conflict Resolution Performance:")
    for name in config.SCENARIOS:
        m = all_metrics[name]
        print(
            f"    {config.SCENARIO_LABELS[name]:30s} "
            f"{m.num_conflicts} conflicts, "
            f"{m.avg_conflict_resolution_ms:.1f}ms avg"
        )

    print()
    print("  Disruption Recovery:")
    for name in config.SCENARIOS:
        m = all_metrics[name]
        print(
            f"    {config.SCENARIO_LABELS[name]:30s} "
            f"{m.disruption_recovery_time:.0f} min recovery, "
            f"{m.schedule_stability:.1%} stability"
        )


def main():
    """Main execution function."""
    print_banner()

    # Check API key
    if config.GROQ_API_KEY == "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":
        print("  ⚠ No Groq API key configured.")
        print("    LLM negotiation will use rule-based fallback.")
        print("    To enable LLM: edit config.py → paste your key from")
        print("    https://console.groq.com/keys")
        print()
    else:
        print("  ✓ Groq API key detected. LLM negotiation enabled.")
        print()

    # Initialize LLM negotiator
    negotiator = LLMNegotiator()

    # Create results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Run all 3 scenarios
    all_metrics: Dict[str, ScenarioMetrics] = {}
    total_start = time.time()

    for scenario in config.SCENARIOS:
        logger.info(f"\nStarting scenario: {config.SCENARIO_LABELS[scenario]}")
        scenario_start = time.time()

        # Create fresh simulation for each scenario
        sim = ConstructionSiteSimulation(
            scenario=scenario,
            llm_negotiator=negotiator if scenario == "nash_llm" else None,
            random_seed=config.RANDOM_SEED,
        )

        # Run simulation
        metrics = sim.run()
        all_metrics[scenario] = metrics

        elapsed = time.time() - scenario_start
        logger.info(
            f"Scenario '{scenario}' completed in {elapsed:.1f}s. "
            f"Completion time: {metrics.total_completion_time:.0f} min"
        )

        # Print individual results
        print_scenario_results(scenario, metrics)

    # Print comparison
    print_comparison(all_metrics)

    # Generate visualizations
    print("\n" + "=" * 60)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 60)

    nash_llm_metrics = all_metrics["nash_llm"]
    negotiation_results = nash_llm_metrics.negotiation_results

    # If no negotiation results from sim, use negotiator's log
    if not negotiation_results and negotiator.negotiation_log:
        negotiation_results = negotiator.negotiation_log

    figure_paths = generate_all_figures(all_metrics, negotiation_results)

    for path in figure_paths:
        print(f"  ✓ Saved: {path}")

    # Save negotiation dialogue log
    negotiator.save_log(config.NEGOTIATION_LOG_PATH)
    print(f"  ✓ Saved: {config.NEGOTIATION_LOG_PATH}")

    # Final summary
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("  SIMULATION COMPLETE")
    print("=" * 60)
    print(f"  Total runtime: {total_elapsed:.1f} seconds")
    print(f"  LLM API calls: {negotiator.total_api_calls}")
    print(f"  Results saved to: {config.RESULTS_DIR}/")
    print()
    print("  Generated files:")
    print(f"    • {config.GANTT_CHART_PATH}")
    print(f"    • {config.COMPLETION_CHART_PATH}")
    print(f"    • {config.RECOVERY_CHART_PATH}")
    print(f"    • {config.RESULTS_DIR}/negotiation_log.png")
    print(f"    • {config.NEGOTIATION_LOG_PATH}")
    print()

    # Key finding
    baseline_t = all_metrics["baseline"].total_completion_time
    system_t = all_metrics["nash_llm"].total_completion_time
    improvement = ((baseline_t - system_t) / baseline_t) * 100

    print("  KEY FINDING:")
    if improvement > 0:
        print(
            f"  The Nash + LLM system reduced completion time by "
            f"{improvement:.1f}% compared to the baseline."
        )
    else:
        print(
            f"  The Nash + LLM system completed construction in "
            f"{system_t:.0f} minutes."
        )
    print(
        f"  LLM negotiation provides transparent, explainable reasoning "
        f"for every resource allocation decision."
    )
    print()


if __name__ == "__main__":
    main()
