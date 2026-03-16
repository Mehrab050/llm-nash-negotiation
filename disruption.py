"""
Disruption Event Handler for Construction Site Simulation.

Manages disruption events that interrupt normal construction operations.
Disruptions include material delays, robot breakdowns, and weather events.

This module provides:
- Disruption event generation and scheduling
- Impact assessment on affected agents
- Recovery tracking and measurement
- Renegotiation triggering

"""

import random
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import config

logger = logging.getLogger(__name__)


@dataclass
class DisruptionEvent:
    """Represents a disruption event on the construction site."""
    event_id: str
    event_type: str           # "material_delay", "robot_breakdown", "weather"
    description: str
    trigger_time: float       # Simulation minutes when disruption fires
    delay_duration: float     # How long the disruption lasts (minutes)
    affected_agents: List[str]  # Agent IDs affected
    affected_resources: List[str]  # Resources made unavailable
    severity: str             # "low", "medium", "high"
    resolved: bool = False
    resolved_time: Optional[float] = None


@dataclass
class RecoveryMetric:
    """Tracks recovery progress after a disruption."""
    disruption_id: str
    time_since_disruption: float   # Minutes since disruption
    schedule_deviation: float       # Minutes behind schedule
    agents_idle: int                # Number of idle agents
    resources_blocked: int          # Number of blocked resources


class DisruptionManager:
    """
    Manages disruption events throughout the construction simulation.

    Handles event generation, impact propagation to agents,
    and recovery measurement for performance comparison.
    """

    def __init__(self, random_seed: int = config.RANDOM_SEED):
        """
        Initialize the disruption manager.

        Args:
            random_seed: Seed for reproducible disruption timing.
        """
        self.rng = random.Random(random_seed)
        self.events: List[DisruptionEvent] = []
        self.recovery_metrics: List[RecoveryMetric] = []
        self.active_disruption: Optional[DisruptionEvent] = None

    def generate_material_delay(self) -> DisruptionEvent:
        """
        Generate a steel material delivery delay disruption.

        The disruption fires between 10:00 AM and 12:00 PM
        (120 to 240 minutes from work start) and delays materials
        by 2 hours.

        Returns:
            DisruptionEvent configured for material delay.
        """
        trigger_time = self.rng.uniform(
            config.DISRUPTION_WINDOW_START,
            config.DISRUPTION_WINDOW_END,
        )

        event = DisruptionEvent(
            event_id="D001",
            event_type=config.DISRUPTION_TYPE,
            description=config.DISRUPTION_DESCRIPTION,
            trigger_time=round(trigger_time, 1),
            delay_duration=config.DISRUPTION_DELAY_MINUTES,
            affected_agents=["A1"],  # Frame Robot needs steel
            affected_resources=["steel_materials"],
            severity="high",
        )

        self.events.append(event)
        logger.info(
            f"Disruption scheduled: {event.description} "
            f"at t={event.trigger_time:.0f}"
        )
        return event

    def activate_disruption(self, event: DisruptionEvent, current_time: float):
        """
        Activate a disruption event — mark it as active and log impact.

        Args:
            event: The disruption to activate.
            current_time: Current simulation time.
        """
        self.active_disruption = event
        clock_time = self._minutes_to_clock(current_time)

        logger.info(
            f"\n{'!'*50}\n"
            f"DISRUPTION ACTIVATED at {clock_time} (t={current_time:.0f})\n"
            f"Type: {event.event_type}\n"
            f"Description: {event.description}\n"
            f"Affected Agents: {event.affected_agents}\n"
            f"Expected Duration: {event.delay_duration:.0f} minutes\n"
            f"{'!'*50}\n"
        )

    def resolve_disruption(self, event: DisruptionEvent, current_time: float):
        """
        Mark a disruption as resolved.

        Args:
            event: The disruption to resolve.
            current_time: Current simulation time.
        """
        event.resolved = True
        event.resolved_time = current_time
        self.active_disruption = None

        clock_time = self._minutes_to_clock(current_time)
        logger.info(
            f"\n  Disruption RESOLVED at {clock_time} (t={current_time:.0f})\n"
            f"  Agents can resume normal operations.\n"
        )

    def record_recovery_metric(
        self,
        disruption_id: str,
        time_since_disruption: float,
        schedule_deviation: float,
        agents_idle: int,
        resources_blocked: int,
    ):
        """
        Record a recovery metric snapshot.

        Called periodically after a disruption to track how quickly
        the system returns to normal operation.
        """
        self.recovery_metrics.append(RecoveryMetric(
            disruption_id=disruption_id,
            time_since_disruption=round(time_since_disruption, 1),
            schedule_deviation=round(schedule_deviation, 1),
            agents_idle=agents_idle,
            resources_blocked=resources_blocked,
        ))

    def calculate_impact(
        self,
        event: DisruptionEvent,
        agents: Dict,
    ) -> Dict:
        """
        Calculate the impact of a disruption on all agents.

        Args:
            event: The active disruption.
            agents: Dict of all ConstructionAgent objects.

        Returns:
            Dict with impact metrics per agent.
        """
        impact = {}
        for agent_id, agent in agents.items():
            is_affected = agent.agent_id in event.affected_agents
            resource_overlap = set(agent.resource_needs) & set(event.affected_resources)

            if is_affected or resource_overlap:
                delay_to_agent = event.delay_duration
                downstream_impact = agent.remaining_work * 0.2  # 20% ripple
            else:
                delay_to_agent = 0
                downstream_impact = agent.remaining_work * 0.05  # Minor ripple

            impact[agent.agent_id] = {
                "agent_name": agent.name,
                "directly_affected": is_affected,
                "delay_minutes": delay_to_agent,
                "downstream_impact_minutes": round(downstream_impact, 1),
                "total_impact": round(delay_to_agent + downstream_impact, 1),
            }

        return impact

    def get_recovery_summary(self) -> Dict:
        """
        Summarize recovery performance after all disruptions.

        Returns:
            Dict with recovery statistics.
        """
        if not self.recovery_metrics:
            return {"status": "no disruptions occurred"}

        deviations = [m.schedule_deviation for m in self.recovery_metrics]
        times = [m.time_since_disruption for m in self.recovery_metrics]

        # Find when schedule deviation drops below 10% of peak
        peak_deviation = max(deviations) if deviations else 0
        recovery_threshold = peak_deviation * 0.1
        recovery_time = None

        for metric in self.recovery_metrics:
            if metric.schedule_deviation <= recovery_threshold and recovery_time is None:
                recovery_time = metric.time_since_disruption

        return {
            "total_disruptions": len(self.events),
            "peak_schedule_deviation": round(peak_deviation, 1),
            "time_to_recover_minutes": round(recovery_time, 1) if recovery_time else None,
            "final_deviation": round(deviations[-1], 1) if deviations else 0,
            "recovery_datapoints": len(self.recovery_metrics),
        }

    @staticmethod
    def _minutes_to_clock(minutes_from_start: float) -> str:
        """Convert simulation minutes to clock time string."""
        hour = config.WORK_START_HOUR + int(minutes_from_start) // 60
        minute = int(minutes_from_start) % 60
        return f"{hour:02d}:{minute:02d}"
