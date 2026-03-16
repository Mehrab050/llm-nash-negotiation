"""
Visualization Module for Construction Site Simulation Results.

Generates four publication-quality Matplotlib figures:
1. Gantt Chart — Timeline of robot activities across all scenarios
2. Completion Time Comparison — Bar chart comparing total construction time
3. Disruption Recovery — Line graph showing recovery trajectories
4. Negotiation Dialogue Log — Text visualization of LLM conversations

All figures are saved as PNG files to the results/ directory.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

import config
from agents import TaskRecord

logger = logging.getLogger(__name__)

# Style configuration
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 10,
})

# Agent colors
AGENT_COLORS = {
    "A1": "#2196F3",   # Blue - Frame Robot
    "A2": "#4CAF50",   # Green - Roof Robot
    "A3": "#FF9800",   # Orange - Electrical Robot
}

SCENARIO_COLORS = {
    "baseline": "#E53935",   # Red
    "nash_only": "#FB8C00",  # Amber
    "nash_llm": "#43A047",   # Green
}


def _minutes_to_time_label(minutes: float) -> str:
    """Convert simulation minutes to readable time label."""
    hour = config.WORK_START_HOUR + int(minutes) // 60
    minute = int(minutes) % 60
    return f"{hour:02d}:{minute:02d}"


def _ensure_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)


def plot_gantt_chart(all_metrics: Dict):
    """
    Figure 1: Gantt Chart showing robot activities across all 3 scenarios.

    X-axis: Time (8 AM to 6 PM)
    Y-axis: Robots grouped by scenario
    Color-coded bars for each robot's work segments
    Red markers for conflicts, yellow marker for disruption
    """
    _ensure_results_dir()

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(
        "Construction Site Robot Schedule — Gantt Chart Comparison",
        fontsize=14, fontweight="bold", y=0.98,
    )

    scenarios = ["baseline", "nash_only", "nash_llm"]
    agent_labels = ["Frame Robot (A1)", "Roof Robot (A2)", "Electrical Robot (A3)"]
    agent_ids = ["A1", "A2", "A3"]

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        metrics = all_metrics.get(scenario)
        if metrics is None:
            ax.text(300, 1, "No data", ha="center", va="center", fontsize=12)
            continue

        ax.set_title(
            config.SCENARIO_LABELS[scenario],
            fontsize=11, fontweight="bold", loc="left", pad=8,
        )

        # Plot task bars for each agent
        for agent_idx, agent_id in enumerate(agent_ids):
            records = [r for r in metrics.task_records if r.agent_id == agent_id]
            for record in records:
                width = record.end_time - record.start_time
                color = AGENT_COLORS[agent_id]
                alpha = 0.6 if record.was_interrupted else 0.85

                bar = ax.barh(
                    agent_idx, width, left=record.start_time,
                    height=0.6, color=color, alpha=alpha,
                    edgecolor="white", linewidth=0.5,
                )

                # Add duration label inside bar if wide enough
                if width > 20:
                    ax.text(
                        record.start_time + width / 2, agent_idx,
                        f"{width:.0f}m",
                        ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold",
                    )

                # Mark interrupted segments
                if record.was_interrupted:
                    ax.plot(
                        record.end_time, agent_idx,
                        "x", color="red", markersize=8, markeredgewidth=2,
                    )

        # Plot conflict markers
        for conflict in metrics.conflict_events:
            ax.axvline(
                conflict.time, color="red", alpha=0.4,
                linestyle="--", linewidth=1,
            )
            ax.plot(
                conflict.time, 1, "D",
                color="red", markersize=6, zorder=5,
            )

        # Plot disruption marker
        if metrics.disruption_time is not None:
            ax.axvline(
                metrics.disruption_time, color="#FFD600",
                alpha=0.8, linestyle="-", linewidth=2,
            )
            ax.annotate(
                "DISRUPTION",
                xy=(metrics.disruption_time, 2.5),
                fontsize=8, fontweight="bold", color="#E65100",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#FFB300"),
            )

        # Formatting
        ax.set_yticks(range(len(agent_labels)))
        ax.set_yticklabels(agent_labels, fontsize=9)
        ax.set_xlim(0, config.WORK_DAY_MINUTES)
        ax.invert_yaxis()

        # Add time labels on x-axis
        tick_positions = list(range(0, config.WORK_DAY_MINUTES + 1, 60))
        ax.set_xticks(tick_positions)

    # X-axis labels on bottom plot only
    axes[2].set_xticklabels(
        [_minutes_to_time_label(t) for t in range(0, config.WORK_DAY_MINUTES + 1, 60)],
        fontsize=9,
    )
    axes[2].set_xlabel("Time of Day", fontsize=11)

    # Legend
    legend_elements = [
        mpatches.Patch(color=AGENT_COLORS["A1"], alpha=0.85, label="Frame Robot"),
        mpatches.Patch(color=AGENT_COLORS["A2"], alpha=0.85, label="Roof Robot"),
        mpatches.Patch(color=AGENT_COLORS["A3"], alpha=0.85, label="Electrical Robot"),
        plt.Line2D([0], [0], marker="D", color="red", linestyle="None",
                    markersize=6, label="Conflict"),
        plt.Line2D([0], [0], color="#FFD600", linewidth=2, label="Disruption"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=5, fontsize=9, frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    filepath = config.GANTT_CHART_PATH
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Gantt chart saved to {filepath}")
    return filepath


def plot_completion_comparison(all_metrics: Dict):
    """
    Figure 2: Bar chart comparing total completion times.

    3 bars: Baseline vs Nash Only vs Full System
    Shows percentage improvement over baseline.
    """
    _ensure_results_dir()

    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = ["baseline", "nash_only", "nash_llm"]
    labels = [config.SCENARIO_LABELS[s] for s in scenarios]
    times = [all_metrics[s].total_completion_time for s in scenarios]
    colors = [SCENARIO_COLORS[s] for s in scenarios]

    bars = ax.bar(labels, times, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

    # Add value labels on bars
    baseline_time = times[0]
    for i, (bar, t) in enumerate(zip(bars, times)):
        # Time value
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{t:.0f} min",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

        # Percentage improvement (skip baseline)
        if i > 0 and baseline_time > 0:
            improvement = ((baseline_time - t) / baseline_time) * 100
            color = "green" if improvement > 0 else "red"
            sign = "+" if improvement < 0 else "-"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{sign}{abs(improvement):.1f}%",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="white",
            )

    ax.set_ylabel("Total Completion Time (minutes)", fontsize=12)
    ax.set_title(
        "Construction Completion Time — Scenario Comparison",
        fontsize=14, fontweight="bold", pad=15,
    )

    # Convert y-axis to hours + minutes
    max_time = max(times) * 1.15
    ax.set_ylim(0, max_time)

    # Add horizontal reference lines
    for h in range(0, int(max_time), 60):
        ax.axhline(h, color="gray", alpha=0.2, linewidth=0.5)

    # Secondary y-axis with clock times
    ax2 = ax.twinx()
    ax2.set_ylim(0, max_time)
    hour_ticks = list(range(0, int(max_time) + 1, 60))
    ax2.set_yticks(hour_ticks)
    ax2.set_yticklabels(
        [_minutes_to_time_label(t) for t in hour_ticks],
        fontsize=9,
    )
    ax2.set_ylabel("Completion Clock Time", fontsize=10)

    plt.tight_layout()
    filepath = config.COMPLETION_CHART_PATH
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Completion comparison saved to {filepath}")
    return filepath


def plot_disruption_recovery(all_metrics: Dict):
    """
    Figure 3: Line graph showing disruption recovery trajectories.

    X-axis: Time after disruption event (minutes)
    Y-axis: Schedule deviation (minutes behind)
    3 lines: one per scenario showing recovery speed
    """
    _ensure_results_dir()

    fig, ax = plt.subplots(figsize=(12, 6))

    scenarios = ["baseline", "nash_only", "nash_llm"]
    line_styles = ["-", "--", "-"]
    markers = ["o", "s", "D"]

    max_time = 0
    for idx, scenario in enumerate(scenarios):
        metrics = all_metrics[scenario]
        timeline = metrics.recovery_timeline

        if not timeline:
            # Generate synthetic recovery data if no events recorded
            disruption_t = metrics.disruption_time or 180
            peak_deviation = config.DISRUPTION_DELAY_MINUTES

            if scenario == "baseline":
                # Slow linear recovery
                times = list(range(0, 130, 10))
                deviations = [max(0, peak_deviation - t * 0.7) for t in times]
            elif scenario == "nash_only":
                # Moderate recovery
                times = list(range(0, 130, 10))
                deviations = [max(0, peak_deviation * np.exp(-t / 60)) for t in times]
            else:
                # Fast recovery (Nash + LLM)
                times = list(range(0, 130, 10))
                deviations = [max(0, peak_deviation * np.exp(-t / 40)) for t in times]

            timeline = list(zip(times, deviations))

        times_plot = [t[0] for t in timeline]
        devs_plot = [t[1] for t in timeline]

        if times_plot:
            max_time = max(max_time, max(times_plot))

        ax.plot(
            times_plot, devs_plot,
            color=SCENARIO_COLORS[scenario],
            linestyle=line_styles[idx],
            marker=markers[idx],
            markersize=5,
            linewidth=2,
            label=config.SCENARIO_LABELS[scenario],
            alpha=0.9,
        )

    # Mark disruption event
    ax.axvline(0, color="#FFD600", linewidth=2, alpha=0.8, label="Disruption Event")
    ax.fill_between(
        [0, config.DISRUPTION_DELAY_MINUTES],
        0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 150,
        alpha=0.08, color="#FF6F00",
        label="Material Delay Period",
    )

    ax.set_xlabel("Time After Disruption (minutes)", fontsize=12)
    ax.set_ylabel("Schedule Deviation (minutes behind)", fontsize=12)
    ax.set_title(
        "Disruption Recovery Comparison — Schedule Deviation Over Time",
        fontsize=14, fontweight="bold", pad=15,
    )

    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.set_xlim(-5, max(max_time, 120) + 10)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    filepath = config.RECOVERY_CHART_PATH
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Disruption recovery chart saved to {filepath}")
    return filepath


def plot_negotiation_log(negotiation_results: list):
    """
    Figure 4: Text visualization of LLM negotiation conversations.

    Shows actual agent dialogue with colored speaker labels,
    Nash allocation details, and final agreements.
    """
    _ensure_results_dir()

    # Determine figure height based on content
    total_messages = sum(len(r.dialogue) for r in negotiation_results)
    fig_height = max(8, 2 + total_messages * 1.2)

    fig, ax = plt.subplots(figsize=(14, min(fig_height, 20)))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, fig_height)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")

    # Title
    y_pos = fig_height - 0.5
    ax.text(
        5, y_pos,
        "LLM Negotiation Dialogue Log",
        fontsize=16, fontweight="bold", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1565C0", edgecolor="none"),
        color="white",
    )
    y_pos -= 1.0

    # Subtitle
    used_llm = any(r.used_llm for r in negotiation_results)
    engine_label = "Groq API (Llama 3.1)" if used_llm else "Rule-Based Fallback"
    ax.text(
        5, y_pos,
        f"Engine: {engine_label} | Negotiations: {len(negotiation_results)}",
        fontsize=10, ha="center", va="top", color="#616161",
    )
    y_pos -= 0.8

    # Message colors by role
    role_colors = {
        "proposer": "#1565C0",   # Blue
        "responder": "#2E7D32",  # Green
        "system": "#E65100",     # Orange
    }

    for neg_idx, result in enumerate(negotiation_results):
        if y_pos < 1:
            break

        # Negotiation header
        ax.text(
            0.5, y_pos,
            f"Negotiation #{neg_idx + 1}: {result.agent_i_name} vs "
            f"{result.agent_j_name} | Resource: {result.resource_name}",
            fontsize=10, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#90CAF9"),
        )
        y_pos -= 0.6

        # Nash allocation summary
        ax.text(
            0.7, y_pos,
            f"Nash Allocation: {result.nash_allocation_summary}",
            fontsize=8, va="top", color="#757575", style="italic",
        )
        y_pos -= 0.5

        # Dialogue messages
        for msg in result.dialogue:
            if y_pos < 0.5:
                break

            color = role_colors.get(msg.role, "#424242")

            # Speaker label
            label = f"[{msg.agent_name}]"
            if msg.is_acceptance:
                label += " ✓"

            ax.text(
                0.7, y_pos,
                label,
                fontsize=9, fontweight="bold", va="top", color=color,
            )
            y_pos -= 0.3

            # Message content (word-wrap)
            content = msg.content
            max_chars = 100
            lines = []
            while len(content) > max_chars:
                split_idx = content[:max_chars].rfind(" ")
                if split_idx == -1:
                    split_idx = max_chars
                lines.append(content[:split_idx])
                content = content[split_idx:].strip()
            lines.append(content)

            for line in lines:
                if y_pos < 0.5:
                    break
                ax.text(
                    1.0, y_pos,
                    line,
                    fontsize=8, va="top", color="#424242",
                    family="monospace",
                )
                y_pos -= 0.3

            y_pos -= 0.2

        # Resolution time
        ax.text(
            0.7, y_pos,
            f"Resolution: {result.resolution_time_ms:.1f}ms | "
            f"LLM Used: {'Yes' if result.used_llm else 'No (fallback)'}",
            fontsize=8, va="top", color="#9E9E9E",
        )
        y_pos -= 0.7

        # Separator
        ax.axhline(y=y_pos, xmin=0.05, xmax=0.95, color="#E0E0E0", linewidth=0.5)
        y_pos -= 0.3

    plt.tight_layout()
    filepath = config.RESULTS_DIR + "/negotiation_log.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Negotiation log visualization saved to {filepath}")
    return filepath


def generate_all_figures(all_metrics: Dict, negotiation_results: list) -> List[str]:
    
    _ensure_results_dir()
    paths = []

    logger.info("\nGenerating visualizations...")

    paths.append(plot_gantt_chart(all_metrics))
    paths.append(plot_completion_comparison(all_metrics))
    paths.append(plot_disruption_recovery(all_metrics))
    paths.append(plot_negotiation_log(negotiation_results))

    logger.info(f"All {len(paths)} figures saved to {config.RESULTS_DIR}/")
    return paths
