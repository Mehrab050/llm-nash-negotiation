# Multi-Agent LLM Negotiation for Robot Resource Allocation Using Nash Bargaining Game Theory

> A decentralized multi-agent system where construction robots negotiate resource conflicts using Nash Bargaining Game Theory and LLM-powered natural language reasoning
---

## Research Motivation

This project is motivated by two important trends in modern robotics and intelligent scheduling:

1. LLM-based Robot Coordination — Large language models (LLMs) are increasingly being explored for real-time decision-making, planning, and coordination in robotic systems. They offer flexible and intelligent control, but a fully centralized setup can limit scalability and create a single point of failure.

2. Game-Theoretic Multi-Agent Scheduling — Game-theoretic scheduling provides a strong framework for coordinating multiple robots that share limited resources. It can improve fairness and efficiency, but these methods often lack clear explanations for why a particular decision was made


## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                CONSTRUCTION SITE SIMULATION              │
│                   (SimPy Environment)                    │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐          │
│  │  Frame   │  │   Roof   │  │  Electrical  │          │
│  │  Robot   │  │  Robot   │  │    Robot     │          │
│  │  (A1)    │  │  (A2)    │  │    (A3)      │          │
│  │ Pri=0.9  │  │ Pri=0.6  │  │  Pri=0.5     │          │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘          │
│       │              │               │                   │
│       └──────────┬───┴───────────────┘                   │
│                  │                                        │
│         ┌────────▼────────┐                              │
│         │ CONFLICT DETECT │                              │
│         └────────┬────────┘                              │
│                  │                                        │
│    ┌─────────────▼──────────────┐                        │
│    │   NASH BARGAINING ENGINE   │                        │
│    │  max (u_i - d_i)(u_j - d_j)│                        │
│    │     SciPy Optimization     │                        │
│    └─────────────┬──────────────┘                        │
│                  │                                        │
│    ┌─────────────▼──────────────┐                        │
│    │  LLM NEGOTIATION LAYER    │                        │
│    │  Each agent argues via    │                        │
│    │  Groq API (Llama 3.1)    │                        │
│    │  Natural language logs    │                        │
│    └─────────────┬──────────────┘                        │
│                  │                                        │
│         ┌────────▼────────┐                              │
│         │   RESOLUTION    │──→ Schedule Updated           │
│         └─────────────────┘                              │
│                                                          │
│  SHARED RESOURCES: Crane(1) | Zone A,B,C | Steel(100)   │
│  DISRUPTION: Material delay triggers renegotiation       │
└─────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/construction-llm-nash-negotiation.git
cd construction-llm-nash-negotiation
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure API Key (Optional but Recommended)

Get a **free** Groq API key:
1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up (free — no credit card required)
3. Navigate to API Keys → Create new key
4. Copy the key

Open `config.py` and paste your key:
```python
GROQ_API_KEY = "gsk_your_key_here"
```

> **Note:** The project works without an API key using rule-based fallback responses. The LLM key enables richer, more realistic negotiation dialogues.

### Step 4: Run
```bash
python main.py
```

## What Each Scenario Shows

The system runs three scenarios for comparison:

| Scenario | Strategy | Intelligence |
|----------|----------|-------------|
| **Baseline (FCFS)** | First-Come-First-Served | None — robots queue blindly |
| **Nash Bargaining Only** | SciPy optimization of Nash product | Mathematical — no language |
| **Nash + LLM (Full System)** | Nash math + LLM arguments | Both — optimal and explainable |

### Scenario A — Baseline
Robots request resources in order. If a resource is busy, they wait in a queue. No intelligence, no optimization. This represents a naive scheduling approach.

### Scenario B — Nash Bargaining Only
When conflicts occur, the Nash Bargaining Engine calculates the mathematically optimal allocation that maximizes the product of both agents' utilities. No natural language — purely mathematical.

### Scenario C — Nash + LLM Negotiation (Our System)
Nash Bargaining calculates the optimal allocation, then each robot's LLM brain generates natural language arguments explaining **why** it accepts or challenges the allocation. This provides:
- **Transparency**: Every decision is explained in plain language
- **Auditability**: Full dialogue logs for review
- **Adaptability**: LLM reasoning considers context that pure math may miss

## Sample Results

When you run the system, you will see:
- **Completion time improvements** of the Nash + LLM system over baseline
- **Gantt chart** showing how each robot's schedule differs across strategies
- **Recovery curves** showing how quickly each strategy recovers from a 2-hour material delay
- **Full negotiation dialogues** between robot agents (the most impressive visual)

## Project Structure

```
construction_llm_negotiation/
├── main.py              # Entry point — run this
├── config.py            # All settings and API keys
├── environment.py       # SimPy construction site simulation
├── agents.py            # Robot agent classes
├── nash_solver.py       # Nash Bargaining SciPy optimizer
├── llm_negotiator.py    # Groq API LLM integration
├── disruption.py        # Disruption event handler
├── visualizer.py        # Matplotlib figure generation
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── results/             # Generated outputs
    ├── gantt_chart.png
    ├── completion_comparison.png
    ├── disruption_recovery.png
    ├── negotiation_log.png
    └── negotiation_log.txt
```

## License

MIT License — free to use, modify, and distribute.
