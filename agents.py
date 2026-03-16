"""
Construction Robot Agent Classes.

Each robot agent has:
- A task priority and urgency profile
- A set of required resources
- A utility function for Nash Bargaining
- An LLM brain for natural language negotiation
- Methods to request, hold, and release resources

Agents operate within a SimPy discrete-event simulation environment.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import config
from nash_solver import AgentUtility
from llm_negotiator import LLMNegotiator

logger = logging.getLogger(__name__)


@dataclass
class TaskRecord:
    """Records a completed or in-progress task segment."""
    agent_id: str
    agent_name: str
    task_name: str
    start_time: float       # Simulation minutes from day start
    end_time: float
    resources_used: List[str]
    was_interrupted: bool = False
    color: str = "#2196F3"


class ConstructionAgent:
    """
    A construction robot agent with LLM-powered negotiation capability.

    Each agent represents a robot on the affordable housing construction
    site. The agent can request resources, detect conflicts, invoke the
    Nash Bargaining solver, and generate LLM negotiation arguments.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        task: str,
        stage: int,
        priority_weight: float,
        deadline_sensitivity: str,
        resource_needs: List[str],
        alternate_zones: Optional[List[str]],
        base_task_duration: float,
        color: str,
    ):
        """
        Initialize a construction robot agent.

        Args:
            agent_id: Unique identifier (e.g., "A1").
            name: Human-readable name (e.g., "Frame Robot").
            task: Description of the agent's construction task.
            stage: Construction stage number (1, 2, or 3).
            priority_weight: Task priority from 0.0 to 1.0.
            deadline_sensitivity: "HIGH", "MEDIUM", or "LOW".
            resource_needs: List of required resource names.
            alternate_zones: Optional list of alternative workspace zones.
            base_task_duration: Total minutes needed for the full task.
            color: Hex color for Gantt chart visualization.
        """
        self.agent_id = agent_id
        self.name = name
        self.task = task
        self.stage = stage
        self.priority_weight = priority_weight
        self.deadline_sensitivity = deadline_sensitivity
        self.resource_needs = resource_needs
        self.alternate_zones = alternate_zones or []
        self.base_task_duration = base_task_duration
        self.color = color

        # State tracking
        self.current_time = 0.0
        self.remaining_work = base_task_duration
        self.is_working = False
        self.is_waiting = False
        self.is_disrupted = False
        self.held_resources: List[str] = []
        self.task_records: List[TaskRecord] = []
        self.conflicts_involved = 0
        self.total_wait_time = 0.0
        self.completion_time: Optional[float] = None

        # Dependency tracking
        self.depends_on_agent: Optional[str] = None
        self.dependency_met = (stage == 1)  # Stage 1 has no dependency
        self.dependency_threshold = 1.0      # Default: wait for full completion

    @classmethod
    def from_config(cls, key: str) -> "ConstructionAgent":
        """
        Create an agent from config.py definitions.

        Args:
            key: Key in config.AGENTS dict (e.g., "frame_robot").

        Returns:
            Initialized ConstructionAgent instance.
        """
        cfg = config.AGENTS[key]
        return cls(
            agent_id=cfg["agent_id"],
            name=cfg["name"],
            task=cfg["task"],
            stage=cfg["stage"],
            priority_weight=cfg["priority_weight"],
            deadline_sensitivity=cfg["deadline_sensitivity"],
            resource_needs=cfg["resource_needs"],
            alternate_zones=cfg.get("alternate_zones", []),
            base_task_duration=cfg["base_task_duration"],
            color=cfg["color"],
        )

    def get_utility_params(self, current_sim_time: float) -> AgentUtility:
        """
        Create an Agent Utility object for Nash Bargaining.

        Calculates time urgency based on how much work remains
        and how much time is left in the working day.

        Args:
            current_sim_time: Current simulation time in minutes.

        Returns:
            AgentUtility for use with NashBargainingSolver.
        """
        # Time urgency: higher when more work remains and less time is left
        time_remaining = max(config.WORK_DAY_MINUTES - current_sim_time, 1)
        work_ratio = self.remaining_work / max(time_remaining, 1)
        time_urgency = min(0.5 + work_ratio * 0.5, 1.0)

        # Critical path factor: stage 1 is most critical
        critical_path_factor = 1.0
        if self.stage == 1:
            critical_path_factor = 1.8
        elif self.stage == 2:
            critical_path_factor = 1.3

        return AgentUtility(
            agent_id=self.agent_id,
            agent_name=self.name,
            task_priority=self.priority_weight,
            time_urgency=time_urgency,
            critical_path_factor=critical_path_factor,
            requested_start=current_sim_time,
            requested_duration=min(self.remaining_work, 60),  # Request up to 1hr chunks
            deadline_sensitivity=self.deadline_sensitivity,
        )

    def get_negotiation_info(self) -> Dict:
        """
        Get agent info dict for the LLM negotiation layer.

        Returns:
            Dict with all info needed for LLM prompt construction.
        """
        return {
            "name": self.name,
            "agent_id": self.agent_id,
            "task": self.task,
            "priority": self.priority_weight,
            "deadline_sensitivity": self.deadline_sensitivity,
            "remaining_work": self.remaining_work,
            "stage": self.stage,
        }

    def record_task(
        self,
        start: float,
        end: float,
        resources: List[str],
        interrupted: bool = False,
    ):
        """
        Record a task execution segment.

        Args:
            start: Start time in simulation minutes.
            end: End time in simulation minutes.
            resources: List of resource names used.
            interrupted: Whether this segment was interrupted.
        """
        self.task_records.append(TaskRecord(
            agent_id=self.agent_id,
            agent_name=self.name,
            task_name=self.task,
            start_time=start,
            end_time=end,
            resources_used=resources,
            was_interrupted=interrupted,
            color=self.color,
        ))

    def work(self, duration: float, current_time: float):
        """
        Simulate performing work for a given duration.

        Args:
            duration: Minutes of work performed.
            current_time: Current simulation time.
        """
        self.remaining_work = max(0, self.remaining_work - duration)
        self.current_time = current_time + duration

        if self.remaining_work <= 0:
            self.completion_time = self.current_time
            logger.info(
                f"  {self.name} COMPLETED all work at t={self.completion_time:.0f}"
            )

    def handle_disruption(self, current_time: float):
        """
        Handle a disruption event — release resources and flag status.

        Args:
            current_time: When the disruption occurred.
        """
        self.is_disrupted = True
        self.is_working = False
        logger.info(
            f"  {self.name} DISRUPTED at t={current_time:.0f}. "
            f"Releasing resources: {self.held_resources}"
        )
        self.held_resources = []

    def resume_after_disruption(self, current_time: float):
        """Resume work after disruption is resolved."""
        self.is_disrupted = False
        logger.info(
            f"  {self.name} RESUMING at t={current_time:.0f}. "
            f"Remaining work: {self.remaining_work:.0f} min"
        )

    def __repr__(self):
        return (
            f"ConstructionAgent({self.agent_id}: {self.name}, "
            f"stage={self.stage}, priority={self.priority_weight}, "
            f"remaining={self.remaining_work:.0f}min)"
        )


def create_all_agents() -> Dict[str, ConstructionAgent]:
    """
    Create all three construction robot agents from config.

    Returns:
        Dict mapping agent keys to ConstructionAgent instances.
    """
    agents = {}
    for key in config.AGENTS:
        agents[key] = ConstructionAgent.from_config(key)

    # Set dependencies with partial overlap thresholds
    # Roof can start when Frame is 60% done (realistic: construction stages overlap)
    # Electrical can start when Roof is 50% done
    agents["roof_robot"].depends_on_agent = "frame_robot"
    agents["roof_robot"].dependency_met = False
    agents["roof_robot"].dependency_threshold = 0.6  # Start when 60% of predecessor done

    agents["electrical_robot"].depends_on_agent = "roof_robot"
    agents["electrical_robot"].dependency_met = False
    agents["electrical_robot"].dependency_threshold = 0.5  # Start when 50% done

    return agents
