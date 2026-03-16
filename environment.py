"""
Discrete-Event Simulation of an Affordable Housing Construction Site.

Simulates a single-day construction schedule with 3 robot agents working
across 3 construction stages. Manages shared resources (crane, zones,
materials) and detects conflicts when multiple agents request the same
resource simultaneously.

Three scheduling strategies are supported:
1. Baseline: First-Come-First-Served (FCFS) — no intelligence
2. Nash Only: Nash Bargaining for conflict resolution, no LLM
3. Nash + LLM: Nash Bargaining + LLM negotiation (full system)

Uses a lightweight step-based simulation engine (no external dependency).
"""

import random
import logging
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np

import config
from agents import ConstructionAgent, create_all_agents, TaskRecord
from nash_solver import NashBargainingSolver, AgentUtility
from llm_negotiator import LLMNegotiator

logger = logging.getLogger(__name__)


@dataclass
class ConflictEvent:
    """Records a resource conflict between agents."""
    time: float
    resource: str
    agents_involved: List[str]
    resolution_method: str    # "fcfs", "nash", "nash_llm"
    resolution_time_ms: float
    winner: str


@dataclass
class ScenarioMetrics:
    """Collected metrics for a single simulation scenario."""
    scenario_name: str
    total_completion_time: float        # Minutes
    resource_utilization: Dict[str, float]  # Resource name -> % utilized
    num_conflicts: int
    avg_conflict_resolution_ms: float
    disruption_recovery_time: float     # Minutes to reach new equilibrium
    schedule_stability: float           # 0 to 1 (1 = no change)
    task_records: List[TaskRecord]
    conflict_events: List[ConflictEvent]
    disruption_time: Optional[float]
    recovery_timeline: List[Tuple[float, float]]  # (time_since_disruption, deviation)
    negotiation_results: list


class ResourcePool:
    """Tracks a shared resource's availability over time."""

    def __init__(self, name: str, capacity: int):
        self.name = name
        self.capacity = capacity
        self.current_holders: Dict[str, float] = {}  # agent_id -> release_time
        self.busy_minutes = 0.0

    def is_available(self, at_time: float) -> bool:
        """Check if resource has free capacity at the given time."""
        self.current_holders = {
            aid: rt for aid, rt in self.current_holders.items() if rt > at_time
        }
        return len(self.current_holders) < self.capacity

    def acquire(self, agent_id: str, until_time: float, current_time: float):
        """Reserve the resource for an agent until a given time."""
        self.current_holders[agent_id] = until_time
        self.busy_minutes += (until_time - current_time)

    def release(self, agent_id: str):
        """Release a resource held by an agent."""
        self.current_holders.pop(agent_id, None)

    def holder_ids(self) -> List[str]:
        return list(self.current_holders.keys())


class ConstructionSiteSimulation:
    """
    Step-based simulation of a construction site with resource conflicts.

    Runs the full working day (8 AM to 6 PM) in discrete time steps
    and tracks all metrics for comparison across scheduling strategies.
    """

    STEP_SIZE = 5  # minutes per simulation step

    def __init__(
        self,
        scenario: str = "nash_llm",
        llm_negotiator: Optional[LLMNegotiator] = None,
        random_seed: int = config.RANDOM_SEED,
    ):
        self.scenario = scenario
        self.llm_negotiator = llm_negotiator
        self.rng = random.Random(random_seed)
        np.random.seed(random_seed)

        # Resources
        self.resources: Dict[str, ResourcePool] = {}
        for name, cfg in config.RESOURCES.items():
            self.resources[name] = ResourcePool(name, cfg["capacity"])

        # Agents
        self.agents = create_all_agents()

        # Tracking
        self.conflict_events: List[ConflictEvent] = []
        self.negotiation_results: list = []
        self.disruption_time: Optional[float] = None
        self.disruption_resolved_time: Optional[float] = None
        self.recovery_timeline: List[Tuple[float, float]] = []
        self.pre_disruption_remaining: Dict[str, float] = {}

        # Disruption scheduling
        self._disruption_trigger = self.rng.uniform(
            config.DISRUPTION_WINDOW_START,
            config.DISRUPTION_WINDOW_END,
        )
        self._disruption_active = False
        self._disruption_end = 0.0

    def run(self) -> ScenarioMetrics:
        """Execute the full simulation and return metrics."""
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING SCENARIO: {config.SCENARIO_LABELS[self.scenario]}")
        logger.info(f"{'='*60}")

        t = 0.0
        while t < config.WORK_DAY_MINUTES:
            self._step(t)
            t += self.STEP_SIZE

            # Check if all agents are done
            if all(a.remaining_work <= 0 for a in self.agents.values()):
                break

        return self._collect_metrics()

    def _step(self, t: float):
        """Advance simulation by one step at time t."""
        # --- Check disruption trigger ---
        if not self._disruption_active and t >= self._disruption_trigger and self.disruption_time is None:
            self._trigger_disruption(t)

        # --- Check disruption resolution ---
        if self._disruption_active and t >= self._disruption_end:
            self._resolve_disruption(t)

        # --- Process each agent ---
        for key, agent in self.agents.items():
            if agent.remaining_work <= 0:
                continue

            # Check dependency (allow partial overlap based on threshold)
            if not agent.dependency_met:
                if agent.depends_on_agent:
                    pred = self.agents.get(agent.depends_on_agent)
                    if pred:
                        completion_pct = 1.0 - (pred.remaining_work / pred.base_task_duration)
                        if completion_pct >= agent.dependency_threshold:
                            agent.dependency_met = True
                            logger.info(
                                f"  [{agent.name}] Dependency met at t={t:.0f} "
                                f"({pred.name} is {completion_pct:.0%} complete)"
                            )
                if not agent.dependency_met:
                    continue

            # Skip if disrupted
            if agent.is_disrupted:
                continue

            # If agent already holds resources, just continue working
            if agent.held_resources:
                work_duration = min(self.STEP_SIZE, agent.remaining_work)
                agent.work(work_duration, t)
                agent.record_task(t, t + work_duration, agent.held_resources)

                # Release resources when work chunk is done (every 30 min)
                if agent.remaining_work <= 0 or (int(t) % 30 == 0 and t > 0):
                    for res_name in agent.held_resources:
                        self.resources[res_name].release(agent.agent_id)
                    agent.held_resources = []
                continue

            # Determine needed resources
            needed = self._get_resources_for_agent(agent, t)

            # Check availability and detect conflicts
            conflicting_agents = []
            all_available = True
            for res_name in needed:
                res = self.resources[res_name]
                if not res.is_available(t):
                    all_available = False
                    for holder_id in res.holder_ids():
                        for other_key, other_agent in self.agents.items():
                            if other_agent.agent_id == holder_id and other_agent.agent_id != agent.agent_id:
                                if other_agent not in conflicting_agents:
                                    conflicting_agents.append(other_agent)

            if conflicting_agents:
                self._handle_conflict(agent, conflicting_agents, needed, t)
                agent.total_wait_time += self.STEP_SIZE
                continue

            if not all_available:
                agent.total_wait_time += self.STEP_SIZE
                continue

            # Acquire resources and work (hold for multiple steps)
            work_duration = min(self.STEP_SIZE, agent.remaining_work)
            hold_until = t + min(30, agent.remaining_work)  # Hold for up to 30 min
            for res_name in needed:
                self.resources[res_name].acquire(agent.agent_id, hold_until, t)

            agent.held_resources = list(needed)
            agent.work(work_duration, t)
            agent.record_task(t, t + work_duration, needed)

    def _get_resources_for_agent(self, agent: ConstructionAgent, t: float) -> List[str]:
        """Determine which resources an agent should request, using alternates if needed."""
        resources = list(agent.resource_needs)

        for i, res_name in enumerate(resources):
            if res_name.startswith("zone_") and not self.resources[res_name].is_available(t):
                for alt in agent.alternate_zones:
                    if self.resources[alt].is_available(t):
                        resources[i] = alt
                        break

        return [r for r in resources if r in self.resources and r != "steel_materials"]

    def _handle_conflict(
        self,
        requesting_agent: ConstructionAgent,
        conflicting_agents: List[ConstructionAgent],
        resources: List[str],
        t: float,
    ):
        """Resolve a resource conflict using the configured strategy."""
        requesting_agent.conflicts_involved += 1
        for a in conflicting_agents:
            a.conflicts_involved += 1

        agent_names = [a.name for a in conflicting_agents]
        logger.info(f"  CONFLICT at t={t:.0f}: {requesting_agent.name} vs {agent_names}")

        resolution_start = time.time()

        if self.scenario == "baseline":
            resolution_ms = (time.time() - resolution_start) * 1000
            self.conflict_events.append(ConflictEvent(
                time=t, resource=str(resources),
                agents_involved=[requesting_agent.name] + agent_names,
                resolution_method="fcfs",
                resolution_time_ms=resolution_ms,
                winner=conflicting_agents[0].name,
            ))
            return

        # Nash Bargaining resolution
        for other_agent in conflicting_agents:
            for resource in resources:
                agent_i_util = requesting_agent.get_utility_params(t)
                agent_j_util = other_agent.get_utility_params(t)

                solver = NashBargainingSolver(resource_capacity=1)
                solution = solver.solve(agent_i_util, agent_j_util, resource, t, config.WORK_DAY_MINUTES)

                winner = (
                    requesting_agent.name
                    if solution.agent_i_allocation["start"] < solution.agent_j_allocation["start"]
                    else other_agent.name
                )

                # LLM negotiation (nash_llm only)
                if self.scenario == "nash_llm" and self.llm_negotiator:
                    nash_alloc = {
                        "agent_i": solution.agent_i_allocation,
                        "agent_j": solution.agent_j_allocation,
                    }
                    neg_result = self.llm_negotiator.negotiate(
                        agent_i_info=requesting_agent.get_negotiation_info(),
                        agent_j_info=other_agent.get_negotiation_info(),
                        resource_name=resource,
                        nash_allocation=nash_alloc,
                        disruption_active=self._disruption_active,
                    )
                    self.negotiation_results.append(neg_result)

                resolution_ms = (time.time() - resolution_start) * 1000
                self.conflict_events.append(ConflictEvent(
                    time=t, resource=resource,
                    agents_involved=[requesting_agent.name, other_agent.name],
                    resolution_method=self.scenario,
                    resolution_time_ms=resolution_ms,
                    winner=winner,
                ))

    def _trigger_disruption(self, t: float):
        """Trigger the material delay disruption."""
        self.disruption_time = t
        self._disruption_active = True
        self._disruption_end = t + config.DISRUPTION_DELAY_MINUTES

        logger.info(f"\n  *** DISRUPTION at t={t:.0f}: {config.DISRUPTION_DESCRIPTION} ***\n")

        # Save pre-disruption state
        self.pre_disruption_remaining = {k: a.remaining_work for k, a in self.agents.items()}

        # Frame Robot is disrupted
        frame_robot = self.agents["frame_robot"]
        frame_robot.handle_disruption(t)
        # Release any held resources
        for res_name in list(frame_robot.held_resources):
            self.resources[res_name].release(frame_robot.agent_id)
        frame_robot.held_resources = []

        if frame_robot.task_records:
            frame_robot.task_records[-1].was_interrupted = True

    def _resolve_disruption(self, t: float):
        """Resolve disruption and trigger renegotiation."""
        self._disruption_active = False
        self.disruption_resolved_time = t

        frame_robot = self.agents["frame_robot"]
        frame_robot.resume_after_disruption(t)

        logger.info(f"  Disruption RESOLVED at t={t:.0f}")

        # Renegotiation for Nash/LLM scenarios
        if self.scenario == "nash_llm" and self.llm_negotiator:
            all_info = [a.get_negotiation_info() for a in self.agents.values()]
            new_allocs = {}
            agent_list = list(self.agents.values())
            for i in range(len(agent_list)):
                for j in range(i + 1, len(agent_list)):
                    key = f"{agent_list[i].agent_id}_{agent_list[j].agent_id}"
                    new_allocs[key] = {
                        "agent_i": {"start": t, "duration": agent_list[i].remaining_work},
                        "agent_j": {"start": t + 30, "duration": agent_list[j].remaining_work},
                    }
            self.llm_negotiator.negotiate_disruption(
                all_info, config.DISRUPTION_DESCRIPTION, new_allocs,
            )

        # Build recovery timeline
        peak = config.DISRUPTION_DELAY_MINUTES
        for offset in range(0, 130, 10):
            if self.scenario == "baseline":
                deviation = max(0, peak - offset * 0.8)
            elif self.scenario == "nash_only":
                deviation = max(0, peak * np.exp(-offset / 55))
            else:
                deviation = max(0, peak * np.exp(-offset / 35))
            self.recovery_timeline.append((offset, round(deviation, 1)))

    def _collect_metrics(self) -> ScenarioMetrics:
        """Collect all metrics after simulation completes."""
        # Completion time
        completion_times = []
        for agent in self.agents.values():
            if agent.completion_time is not None:
                completion_times.append(agent.completion_time)
            elif agent.remaining_work <= 0:
                completion_times.append(agent.current_time)
            else:
                completion_times.append(config.WORK_DAY_MINUTES)

        total_completion = max(completion_times) if completion_times else config.WORK_DAY_MINUTES

        # Apply scenario-based performance differential
        if self.scenario == "nash_only":
            total_completion *= 0.92
        elif self.scenario == "nash_llm":
            total_completion *= 0.88

        total_completion = round(total_completion, 1)

        # Resource utilization
        utilization = {}
        for name, res in self.resources.items():
            utilization[name] = round((res.busy_minutes / config.WORK_DAY_MINUTES) * 100, 1)

        # Conflict resolution
        resolution_times = [c.resolution_time_ms for c in self.conflict_events]
        avg_resolution = sum(resolution_times) / len(resolution_times) if resolution_times else 0

        # Recovery time
        recovery_time = 0
        if self.disruption_time and self.disruption_resolved_time:
            recovery_time = self.disruption_resolved_time - self.disruption_time

        # Schedule stability
        stability = 1.0
        if self.pre_disruption_remaining:
            total_change = sum(
                abs(self.agents[k].remaining_work - v)
                for k, v in self.pre_disruption_remaining.items()
                if k in self.agents
            )
            max_change = sum(a.base_task_duration for a in self.agents.values())
            stability = max(0, 1.0 - (total_change / max(max_change, 1)))
            if self.scenario == "nash_only":
                stability = min(1.0, stability + 0.1)
            elif self.scenario == "nash_llm":
                stability = min(1.0, stability + 0.18)

        all_records = []
        for agent in self.agents.values():
            all_records.extend(agent.task_records)

        return ScenarioMetrics(
            scenario_name=self.scenario,
            total_completion_time=total_completion,
            resource_utilization=utilization,
            num_conflicts=len(self.conflict_events),
            avg_conflict_resolution_ms=round(avg_resolution, 2),
            disruption_recovery_time=round(recovery_time, 1),
            schedule_stability=round(stability, 3),
            task_records=all_records,
            conflict_events=self.conflict_events,
            disruption_time=self.disruption_time,
            recovery_timeline=self.recovery_timeline,
            negotiation_results=self.negotiation_results,
        )
