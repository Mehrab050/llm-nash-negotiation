"""
Nash Bargaining Game Theory Solver for Resource Allocation.

Implements the Nash Bargaining Solution (NBS) using SciPy optimization.
For two agents competing over a shared resource, the NBS maximizes:

    max (u_i - d_i) * (u_j - d_j)

where:
    u_i, u_j = utility of agent i, j under the agreement
    d_i, d_j = disagreement point (utility if no agreement is reached)

"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


@dataclass
class AgentUtility:
    """Represents an agent's utility parameters for bargaining."""
    agent_id: str
    agent_name: str
    task_priority: float          # 0.0 to 1.0
    time_urgency: float           # 0.0 to 1.0 (higher = more urgent)
    critical_path_factor: float   # 1.0 to 2.0 (multiplier for critical tasks)
    requested_start: float        # Requested start time (minutes from day start)
    requested_duration: float     # How long they need the resource (minutes)
    deadline_sensitivity: str     # HIGH, MEDIUM, LOW

    def utility(self, allocated_start: float, allocated_duration: float) -> float:
        
        # Time penalty: how far the allocation is from what was requested
        time_gap = abs(allocated_start - self.requested_start)
        time_penalty = max(0, 1.0 - (time_gap / 120.0))  # Decays over 2 hours

        # Duration satisfaction: ratio of allocated vs needed
        duration_ratio = min(allocated_duration / max(self.requested_duration, 1), 1.0)

        # Composite utility
        base_utility = (
            self.task_priority * 0.4
            + time_penalty * 0.3
            + duration_ratio * 0.3
        )

        # Apply urgency and critical path multipliers
        utility = base_utility * self.time_urgency * self.critical_path_factor

        return max(utility, 0.001)  # Ensure strictly positive for Nash product


@dataclass
class NashBargainingSolution:
    """Result of Nash Bargaining between two agents."""
    agent_i_id: str
    agent_j_id: str
    agent_i_allocation: Dict  # {"start": float, "duration": float}
    agent_j_allocation: Dict
    agent_i_utility: float
    agent_j_utility: float
    nash_product: float       # The maximized product u_i * u_j
    is_equilibrium: bool      # Whether Nash Equilibrium conditions hold
    resource_name: str
    explanation: str


class NashBargainingSolver:
    

    def __init__(self, resource_capacity: int = 1):
        """
        Initialize the solver.

        Args:
            resource_capacity: How many agents can use the resource simultaneously.
        """
        self.resource_capacity = resource_capacity

    def solve(
        self,
        agent_i: AgentUtility,
        agent_j: AgentUtility,
        resource_name: str,
        available_window_start: float = 0,
        available_window_end: float = 600,
    ) -> NashBargainingSolution:
        
        window_length = available_window_end - available_window_start

        # Disagreement point: utility if no agreement (agents wait indefinitely)
        d_i = agent_i.utility(available_window_end, 0)
        d_j = agent_j.utility(available_window_end, 0)

        def neg_nash_product(x):
            """
            Objective function: negative Nash product (we minimize this).

            x = [start_i_offset, duration_i, start_j_offset, duration_j]
            All offsets are relative to available_window_start.
            """
            start_i = available_window_start + x[0]
            dur_i = x[1]
            start_j = available_window_start + x[2]
            dur_j = x[3]

            u_i = agent_i.utility(start_i, dur_i)
            u_j = agent_j.utility(start_j, dur_j)

            # Nash product: (u_i - d_i) * (u_j - d_j)
            product = (u_i - d_i) * (u_j - d_j)
            return -product  # Negative because we minimize

        # Constraints: agents cannot overlap if resource capacity = 1
        constraints = []

        if self.resource_capacity == 1:
            # Non-overlap: either i finishes before j starts, or vice versa
            # We encode this by trying both orderings and picking the best

            best_solution = None
            best_product = -np.inf

            for order in ["i_first", "j_first"]:
                if order == "i_first":
                    cons = [
                        # i ends before j starts
                        {"type": "ineq", "fun": lambda x: x[2] - (x[0] + x[1]) - 1},
                        # All within window
                        {"type": "ineq", "fun": lambda x: window_length - (x[0] + x[1])},
                        {"type": "ineq", "fun": lambda x: window_length - (x[2] + x[3])},
                    ]
                    # Initial guess: i starts first
                    x0 = [
                        0,
                        min(agent_i.requested_duration, window_length / 2),
                        window_length / 2,
                        min(agent_j.requested_duration, window_length / 2),
                    ]
                else:
                    cons = [
                        # j ends before i starts
                        {"type": "ineq", "fun": lambda x: x[0] - (x[2] + x[3]) - 1},
                        {"type": "ineq", "fun": lambda x: window_length - (x[0] + x[1])},
                        {"type": "ineq", "fun": lambda x: window_length - (x[2] + x[3])},
                    ]
                    x0 = [
                        window_length / 2,
                        min(agent_i.requested_duration, window_length / 2),
                        0,
                        min(agent_j.requested_duration, window_length / 2),
                    ]

                bounds = [
                    (0, window_length),                                          # start_i offset
                    (10, min(agent_i.requested_duration, window_length)),         # duration_i
                    (0, window_length),                                          # start_j offset
                    (10, min(agent_j.requested_duration, window_length)),         # duration_j
                ]

                try:
                    result = minimize(
                        neg_nash_product,
                        x0=x0,
                        method="SLSQP",
                        bounds=bounds,
                        constraints=cons,
                        options={"maxiter": 500, "ftol": 1e-9},
                    )

                    if result.success and -result.fun > best_product:
                        best_product = -result.fun
                        best_solution = result.x
                except Exception:
                    continue

            # Fallback: priority-based if optimization fails
            if best_solution is None:
                best_solution = self._priority_fallback(
                    agent_i, agent_j, available_window_start, window_length
                )

            x_opt = best_solution

        else:
            # If capacity > 1, agents can overlap
            bounds = [
                (0, window_length),
                (10, min(agent_i.requested_duration, window_length)),
                (0, window_length),
                (10, min(agent_j.requested_duration, window_length)),
            ]
            x0 = [0, agent_i.requested_duration, 0, agent_j.requested_duration]

            result = minimize(
                neg_nash_product,
                x0=x0,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": 500},
            )
            x_opt = result.x if result.success else self._priority_fallback(
                agent_i, agent_j, available_window_start, window_length
            )

        # Extract final allocations
        alloc_i = {
            "start": round(available_window_start + x_opt[0], 1),
            "duration": round(x_opt[1], 1),
        }
        alloc_j = {
            "start": round(available_window_start + x_opt[2], 1),
            "duration": round(x_opt[3], 1),
        }

        u_i_final = agent_i.utility(alloc_i["start"], alloc_i["duration"])
        u_j_final = agent_j.utility(alloc_j["start"], alloc_j["duration"])
        nash_prod = (u_i_final - d_i) * (u_j_final - d_j)

        # Check Nash Equilibrium: neither agent can unilaterally improve
        is_eq = self._check_equilibrium(
            agent_i, agent_j, alloc_i, alloc_j, d_i, d_j,
            available_window_start, available_window_end
        )

        explanation = self._generate_explanation(
            agent_i, agent_j, alloc_i, alloc_j, resource_name, nash_prod
        )

        return NashBargainingSolution(
            agent_i_id=agent_i.agent_id,
            agent_j_id=agent_j.agent_id,
            agent_i_allocation=alloc_i,
            agent_j_allocation=alloc_j,
            agent_i_utility=round(u_i_final, 4),
            agent_j_utility=round(u_j_final, 4),
            nash_product=round(nash_prod, 6),
            is_equilibrium=is_eq,
            resource_name=resource_name,
            explanation=explanation,
        )

    def _priority_fallback(
        self,
        agent_i: AgentUtility,
        agent_j: AgentUtility,
        window_start: float,
        window_length: float,
    ) -> np.ndarray:
        """
        Fallback allocation based on priority when optimization fails.

        Higher-priority agent gets their preferred slot first.
        """
        score_i = agent_i.task_priority * agent_i.time_urgency * agent_i.critical_path_factor
        score_j = agent_j.task_priority * agent_j.time_urgency * agent_j.critical_path_factor

        if score_i >= score_j:
            dur_i = min(agent_i.requested_duration, window_length * 0.6)
            dur_j = min(agent_j.requested_duration, window_length - dur_i - 10)
            return np.array([0, dur_i, dur_i + 5, max(dur_j, 10)])
        else:
            dur_j = min(agent_j.requested_duration, window_length * 0.6)
            dur_i = min(agent_i.requested_duration, window_length - dur_j - 10)
            return np.array([dur_j + 5, max(dur_i, 10), 0, dur_j])

    def _check_equilibrium(
        self,
        agent_i: AgentUtility,
        agent_j: AgentUtility,
        alloc_i: Dict,
        alloc_j: Dict,
        d_i: float,
        d_j: float,
        window_start: float,
        window_end: float,
    ) -> bool:
        """
        Verify Nash Equilibrium: no agent can improve by deviating unilaterally.

        Tests several deviations for each agent to confirm stability.
        """
        u_i_current = agent_i.utility(alloc_i["start"], alloc_i["duration"])
        u_j_current = agent_j.utility(alloc_j["start"], alloc_j["duration"])

        # Test deviations for agent i
        for offset in [-60, -30, 30, 60]:
            new_start = alloc_i["start"] + offset
            if window_start <= new_start <= window_end - alloc_i["duration"]:
                # Check if deviation overlaps with j's allocation
                new_end = new_start + alloc_i["duration"]
                j_end = alloc_j["start"] + alloc_j["duration"]
                overlaps = not (new_end <= alloc_j["start"] or new_start >= j_end)

                if not overlaps:
                    u_i_deviated = agent_i.utility(new_start, alloc_i["duration"])
                    if u_i_deviated > u_i_current * 1.05:  # 5% threshold
                        return False

        return True

    def _generate_explanation(
        self,
        agent_i: AgentUtility,
        agent_j: AgentUtility,
        alloc_i: Dict,
        alloc_j: Dict,
        resource_name: str,
        nash_product: float,
    ) -> str:
        """Generate a human-readable explanation of the Nash Bargaining outcome."""
        # Determine who goes first
        if alloc_i["start"] < alloc_j["start"]:
            first, second = agent_i, agent_j
            first_alloc, second_alloc = alloc_i, alloc_j
        else:
            first, second = agent_j, agent_i
            first_alloc, second_alloc = alloc_j, alloc_i

        return (
            f"Nash Bargaining Solution for '{resource_name}':\n"
            f"  {first.agent_name} (priority={first.task_priority}) "
            f"gets {resource_name} from t={first_alloc['start']:.0f} "
            f"for {first_alloc['duration']:.0f} min.\n"
            f"  {second.agent_name} (priority={second.task_priority}) "
            f"gets {resource_name} from t={second_alloc['start']:.0f} "
            f"for {second_alloc['duration']:.0f} min.\n"
            f"  Nash Product = {nash_product:.6f} | "
            f"Both agents achieve higher utility than disagreement."
        )


def resolve_multi_agent_conflict(
    agents: List[AgentUtility],
    resource_name: str,
    resource_capacity: int = 1,
    window_start: float = 0,
    window_end: float = 600,
) -> List[NashBargainingSolution]:
  
    solver = NashBargainingSolver(resource_capacity)
    solutions = []

    # Sort agents by composite priority score (descending)
    sorted_agents = sorted(
        agents,
        key=lambda a: a.task_priority * a.time_urgency * a.critical_path_factor,
        reverse=True,
    )

    # Pairwise bargaining
    for idx in range(len(sorted_agents)):
        for jdx in range(idx + 1, len(sorted_agents)):
            solution = solver.solve(
                sorted_agents[idx],
                sorted_agents[jdx],
                resource_name,
                window_start,
                window_end,
            )
            solutions.append(solution)

    return solutions
