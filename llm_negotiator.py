"""
LLM Negotiation Layer using Groq API (free tier).

Each construction robot agent gets its own LLM call to generate
natural language negotiation arguments. This provides an explainability
and transparency layer on top of the Nash Bargaining mathematical solution.

Uses Groq's free API with Llama 3.1 or Mixtral models.
Falls back to rule-based responses if API is unavailable.
"""

import time
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import config

logger = logging.getLogger(__name__)

# Try to import groq; set flag for offline fallback
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq package not installed. Using rule-based fallback.")


@dataclass
class NegotiationMessage:
    """A single message in a negotiation dialogue."""
    timestamp: str
    agent_id: str
    agent_name: str
    role: str           # "proposer", "responder", "system"
    content: str
    is_acceptance: bool = False


@dataclass
class NegotiationResult:
    """Complete result of a negotiation between two agents."""
    resource_name: str
    agent_i_name: str
    agent_j_name: str
    dialogue: list       # List of NegotiationMessage
    final_agreement: str
    resolution_time_ms: float
    used_llm: bool
    nash_allocation_summary: str


class LLMNegotiator:
    """
    Manages LLM-powered negotiation between construction robot agents.

    Each agent generates natural language arguments based on:
    - Their task description and priority
    - The Nash Bargaining suggested allocation
    - Downstream task dependencies
    - Current disruption state
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the negotiator with Groq API.

        Args:
            api_key: Groq API key. If None, uses config value.
        """
        self.api_key = api_key or config.GROQ_API_KEY
        self.client = None
        self.negotiation_log: list = []
        self.total_api_calls = 0

        if GROQ_AVAILABLE and self.api_key != "YOUR_GROQ_API_KEY_HERE":
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info("Groq API client initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to init Groq client: {e}. Using fallback.")
                self.client = None
        else:
            if not GROQ_AVAILABLE:
                logger.info("Groq package not available. Using rule-based negotiation.")
            else:
                logger.info("No API key configured. Using rule-based negotiation.")

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Make a single LLM API call via Groq.

        Args:
            system_prompt: System role instruction.
            user_prompt: The negotiation scenario for the agent.

        Returns:
            LLM response text, or fallback if API fails.
        """
        if self.client is None:
            return self._rule_based_fallback(user_prompt)

        try:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
            )
            self.total_api_calls += 1
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"Groq API call failed: {e}. Using fallback.")
            # Try fallback model
            try:
                response = self.client.chat.completions.create(
                    model=config.LLM_FALLBACK_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=config.LLM_MAX_TOKENS,
                    temperature=config.LLM_TEMPERATURE,
                )
                self.total_api_calls += 1
                return response.choices[0].message.content.strip()
            except Exception as e2:
                logger.warning(f"Fallback model also failed: {e2}")
                return self._rule_based_fallback(user_prompt)

    def _rule_based_fallback(self, prompt: str) -> str:
        """
        Generate a deterministic negotiation response when LLM is unavailable.

        Parses the prompt for key terms and constructs a reasonable response.
        """
        prompt_lower = prompt.lower()

        if "high" in prompt_lower and "urgency" in prompt_lower:
            return (
                "As a high-priority agent on the critical path, I need this "
                "resource urgently. Any delay to my task cascades to all "
                "downstream operations. I accept the Nash allocation that "
                "gives me earlier access."
            )
        elif "accept" in prompt_lower or "agree" in prompt_lower:
            return (
                "I understand the mathematical fairness of this allocation. "
                "While I would prefer earlier access, the Nash solution "
                "ensures both agents can complete their tasks. I agree to "
                "the proposed schedule."
            )
        elif "disruption" in prompt_lower or "delay" in prompt_lower:
            return (
                "Given the material delay disruption, I recognize we need "
                "to renegotiate. I am releasing my current resources and "
                "request priority based on updated urgency scores. Let us "
                "find a new optimal allocation."
            )
        else:
            return (
                "Based on my task priority and timeline requirements, "
                "I propose we follow the Nash Bargaining allocation. "
                "This ensures a fair division of resource access time."
            )

    def negotiate(
        self,
        agent_i_info: Dict,
        agent_j_info: Dict,
        resource_name: str,
        nash_allocation: Dict,
        disruption_active: bool = False,
    ) -> NegotiationResult:
        """
        Run a full negotiation dialogue between two agents.

        Process:
        1. Agent i proposes based on Nash allocation
        2. Agent j responds (accept/counter)
        3. If counter, agent i responds
        4. Final agreement reached (Nash solution is binding)

        Args:
            agent_i_info: Dict with keys: name, agent_id, task, priority,
                          urgency, resource_needs, deadline_sensitivity
            agent_j_info: Same structure for competing agent.
            resource_name: The contested resource.
            nash_allocation: Dict with agent allocations from Nash solver.
            disruption_active: Whether a disruption is currently active.

        Returns:
            NegotiationResult with full dialogue and resolution metadata.
        """
        start_time = time.time()
        dialogue = []
        used_llm = self.client is not None
        now_str = datetime.now().strftime("%H:%M:%S")

        # Format Nash allocation for prompts
        alloc_i = nash_allocation.get("agent_i", {})
        alloc_j = nash_allocation.get("agent_j", {})

        def _minutes_to_time(minutes_from_start):
            """Convert simulation minutes to clock time string."""
            hour = config.WORK_START_HOUR + int(minutes_from_start) // 60
            minute = int(minutes_from_start) % 60
            return f"{hour:02d}:{minute:02d}"

        alloc_i_time = _minutes_to_time(alloc_i.get("start", 0))
        alloc_i_dur = alloc_i.get("duration", 0)
        alloc_j_time = _minutes_to_time(alloc_j.get("start", 0))
        alloc_j_dur = alloc_j.get("duration", 0)

        disruption_context = ""
        if disruption_active:
            disruption_context = (
                " ALERT: A material delivery disruption is currently active. "
                "Steel materials are delayed by 2 hours. Schedules must be "
                "renegotiated to minimize project delay."
            )

        # --- System prompt for all agents ---
        system_prompt = (
            "You are a construction robot AI agent on an affordable housing "
            "construction site. You must negotiate resource access with another "
            "robot agent. Be concise (2-3 sentences). Consider task priority, "
            "project timeline, and downstream dependencies. The Nash Bargaining "
            "mathematical solution provides the optimal allocation — argue for "
            "why you accept or propose an adjustment."
        )

        # --- Round 1: Agent i proposes ---
        prompt_i = (
            f"You are {agent_i_info['name']} ({agent_i_info['agent_id']}). "
            f"Your task: {agent_i_info['task']}. "
            f"Priority: {agent_i_info['priority']}, "
            f"Urgency: {agent_i_info['deadline_sensitivity']}. "
            f"You need '{resource_name}' starting at {alloc_i_time} "
            f"for {alloc_i_dur:.0f} minutes. "
            f"{agent_j_info['name']} also needs '{resource_name}' "
            f"starting at {alloc_j_time} for {alloc_j_dur:.0f} minutes. "
            f"The Nash Bargaining calculation allocates you the slot at "
            f"{alloc_i_time}.{disruption_context} "
            f"Generate a 2-3 sentence negotiation argument for your position."
        )

        response_i = self._call_llm(system_prompt, prompt_i)
        dialogue.append(NegotiationMessage(
            timestamp=now_str,
            agent_id=agent_i_info["agent_id"],
            agent_name=agent_i_info["name"],
            role="proposer",
            content=response_i,
        ))

        # --- Round 2: Agent j responds ---
        prompt_j = (
            f"You are {agent_j_info['name']} ({agent_j_info['agent_id']}). "
            f"Your task: {agent_j_info['task']}. "
            f"Priority: {agent_j_info['priority']}, "
            f"Urgency: {agent_j_info['deadline_sensitivity']}. "
            f"{agent_i_info['name']} just said: \"{response_i}\" "
            f"The Nash allocation gives you the slot at {alloc_j_time} "
            f"for {alloc_j_dur:.0f} minutes.{disruption_context} "
            f"Respond in 2-3 sentences: accept the allocation or explain "
            f"your counter-argument."
        )

        response_j = self._call_llm(system_prompt, prompt_j)
        is_accept_j = any(
            w in response_j.lower()
            for w in ["accept", "agree", "understood", "fair", "acknowledge"]
        )
        dialogue.append(NegotiationMessage(
            timestamp=now_str,
            agent_id=agent_j_info["agent_id"],
            agent_name=agent_j_info["name"],
            role="responder",
            content=response_j,
            is_acceptance=is_accept_j,
        ))

        # --- Round 3: If not accepted, agent i confirms (Nash is binding) ---
        if not is_accept_j:
            prompt_i_final = (
                f"You are {agent_i_info['name']}. "
                f"{agent_j_info['name']} responded: \"{response_j}\" "
                f"The Nash Bargaining Solution is mathematically optimal and "
                f"binding. Acknowledge their concern and confirm the allocation "
                f"in 2 sentences."
            )
            response_i_final = self._call_llm(system_prompt, prompt_i_final)
            dialogue.append(NegotiationMessage(
                timestamp=now_str,
                agent_id=agent_i_info["agent_id"],
                agent_name=agent_i_info["name"],
                role="proposer",
                content=response_i_final,
                is_acceptance=True,
            ))

        # --- System: Agreement reached ---
        final_msg = (
            f"AGREEMENT REACHED: {agent_i_info['name']} gets '{resource_name}' "
            f"at {alloc_i_time} for {alloc_i_dur:.0f}min. "
            f"{agent_j_info['name']} gets '{resource_name}' at {alloc_j_time} "
            f"for {alloc_j_dur:.0f}min. Nash Equilibrium confirmed."
        )
        dialogue.append(NegotiationMessage(
            timestamp=now_str,
            agent_id="SYS",
            agent_name="System",
            role="system",
            content=final_msg,
            is_acceptance=True,
        ))

        elapsed_ms = (time.time() - start_time) * 1000

        result = NegotiationResult(
            resource_name=resource_name,
            agent_i_name=agent_i_info["name"],
            agent_j_name=agent_j_info["name"],
            dialogue=dialogue,
            final_agreement=final_msg,
            resolution_time_ms=round(elapsed_ms, 2),
            used_llm=used_llm,
            nash_allocation_summary=(
                f"{agent_i_info['name']}: start={alloc_i_time}, "
                f"dur={alloc_i_dur:.0f}min | "
                f"{agent_j_info['name']}: start={alloc_j_time}, "
                f"dur={alloc_j_dur:.0f}min"
            ),
        )

        self.negotiation_log.append(result)
        return result

    def negotiate_disruption(
        self,
        all_agents_info: list,
        disruption_description: str,
        new_nash_allocations: Dict,
    ) -> list:
        """
        Handle renegotiation after a disruption event.

        All agents receive disruption notification and renegotiate
        their resource allocations.

        Args:
            all_agents_info: List of agent info dicts.
            disruption_description: What happened.
            new_nash_allocations: Updated Nash allocations post-disruption.

        Returns:
            List of NegotiationResult for each pairwise renegotiation.
        """
        results = []
        now_str = datetime.now().strftime("%H:%M:%S")

        # First: disruption announcement from affected agent
        announcement = NegotiationMessage(
            timestamp=now_str,
            agent_id=all_agents_info[0]["agent_id"],
            agent_name=all_agents_info[0]["name"],
            role="system",
            content=(
                f"DISRUPTION ALERT: {disruption_description}. "
                f"I ({all_agents_info[0]['name']}) am releasing held resources. "
                f"All agents must renegotiate schedules immediately."
            ),
        )

        # Log the announcement
        self.negotiation_log.append(
            NegotiationResult(
                resource_name="ALL",
                agent_i_name=all_agents_info[0]["name"],
                agent_j_name="ALL",
                dialogue=[announcement],
                final_agreement="Renegotiation triggered",
                resolution_time_ms=0,
                used_llm=False,
                nash_allocation_summary="Disruption — recalculating...",
            )
        )

        # Pairwise renegotiation with disruption context
        for i in range(len(all_agents_info)):
            for j in range(i + 1, len(all_agents_info)):
                key = f"{all_agents_info[i]['agent_id']}_{all_agents_info[j]['agent_id']}"
                alloc = new_nash_allocations.get(key, {})

                result = self.negotiate(
                    agent_i_info=all_agents_info[i],
                    agent_j_info=all_agents_info[j],
                    resource_name="shared_resources",
                    nash_allocation=alloc,
                    disruption_active=True,
                )
                results.append(result)

        return results

    def save_log(self, filepath: str):
        """
        Save all negotiation dialogues to a text file.

        Args:
            filepath: Path to output file.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("NEGOTIATION DIALOGUE LOG\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total API calls: {self.total_api_calls}\n")
            f.write("=" * 70 + "\n\n")

            for idx, result in enumerate(self.negotiation_log):
                f.write(f"--- Negotiation #{idx + 1} ---\n")
                f.write(f"Resource: {result.resource_name}\n")
                f.write(f"Parties: {result.agent_i_name} vs {result.agent_j_name}\n")
                f.write(f"Resolution Time: {result.resolution_time_ms:.1f} ms\n")
                f.write(f"Used LLM: {result.used_llm}\n")
                f.write(f"Nash Allocation: {result.nash_allocation_summary}\n\n")

                for msg in result.dialogue:
                    prefix = f"[{msg.timestamp}] {msg.agent_name} ({msg.role})"
                    f.write(f"{prefix}:\n")
                    f.write(f"  {msg.content}\n\n")

                f.write(f"FINAL: {result.final_agreement}\n")
                f.write("-" * 70 + "\n\n")

        logger.info(f"Negotiation log saved to {filepath}")
