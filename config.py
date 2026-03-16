"""
Configuration file for the Multi-Agent LLM Negotiation System.
Contains all tunable parameters, API keys, and simulation settings.
"""

# =============================================================================
# GROQ API CONFIGURATION
# =============================================================================
# Get your free API key at: https://console.groq.com/keys
# The free tier gives 30 requests/minute which is plenty for this project.
GROQ_API_KEY = "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"

# LLM model to use (free on Groq)
LLM_MODEL = "llama-3.1-8b-instant"

# Fallback model if primary is unavailable
LLM_FALLBACK_MODEL = "mixtral-8x7b-32768"

# Maximum tokens per LLM response
LLM_MAX_TOKENS = 256

# Temperature for LLM responses (lower = more deterministic)
LLM_TEMPERATURE = 0.7

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
# Working day: 8:00 AM to 6:00 PM = 10 hours = 600 minutes
WORK_START_HOUR = 8       # 8:00 AM
WORK_END_HOUR = 18        # 6:00 PM
WORK_DAY_MINUTES = 600    # Total working minutes

# SimPy simulation speed (1 sim unit = 1 minute of construction time)
SIM_TIME_UNIT = "minutes"

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# AGENT DEFINITIONS
# =============================================================================
AGENTS = {
    "frame_robot": {
        "name": "Frame Robot",
        "agent_id": "A1",
        "task": "Assemble wall frames and structural components",
        "stage": 1,
        "priority_weight": 0.9,
        "deadline_sensitivity": "HIGH",
        "resource_needs": ["zone_a", "crane", "steel_materials"],
        "base_task_duration": 180,  # minutes for full task
        "color": "#2196F3",         # Blue for Gantt chart
    },
    "roof_robot": {
        "name": "Roof Robot",
        "agent_id": "A2",
        "task": "Install roof panels and insulation",
        "stage": 2,
        "priority_weight": 0.6,
        "deadline_sensitivity": "MEDIUM",
        "resource_needs": ["zone_a", "crane"],  # Can also use zone_b
        "alternate_zones": ["zone_b"],
        "base_task_duration": 150,  # minutes
        "color": "#4CAF50",         # Green for Gantt chart
    },
    "electrical_robot": {
        "name": "Electrical Robot",
        "agent_id": "A3",
        "task": "Run wiring and install electrical components",
        "stage": 3,
        "priority_weight": 0.5,
        "deadline_sensitivity": "MEDIUM",
        "resource_needs": ["zone_b"],  # Can also use zone_c, no crane
        "alternate_zones": ["zone_c"],
        "base_task_duration": 120,  # minutes
        "color": "#FF9800",         # Orange for Gantt chart
    },
}

# =============================================================================
# RESOURCE DEFINITIONS
# =============================================================================
RESOURCES = {
    "crane": {
        "capacity": 1,
        "description": "Main construction crane (only one on site)",
    },
    "zone_a": {
        "capacity": 1,
        "description": "Primary workspace zone",
    },
    "zone_b": {
        "capacity": 1,
        "description": "Secondary workspace zone",
    },
    "zone_c": {
        "capacity": 1,
        "description": "Tertiary workspace zone",
    },
    "steel_materials": {
        "capacity": 100,
        "description": "Daily steel material supply (units)",
    },
}

# =============================================================================
# DISRUPTION SETTINGS
# =============================================================================
# Disruption window: between 10:00 AM and 12:00 PM
# In simulation minutes from start: 120 to 240 (since work starts at 8 AM)
DISRUPTION_WINDOW_START = 120   # minutes after work start (10:00 AM)
DISRUPTION_WINDOW_END = 240     # minutes after work start (12:00 PM)
DISRUPTION_TYPE = "material_delay"
DISRUPTION_DELAY_MINUTES = 120  # 2-hour material delay
DISRUPTION_DESCRIPTION = "Steel materials delivery delayed by 2 hours"

# =============================================================================
# CONSTRUCTION STAGES
# =============================================================================
STAGES = {
    1: {
        "name": "Foundation & Frame Assembly",
        "description": "Structural framing and foundation work",
        "primary_agent": "frame_robot",
    },
    2: {
        "name": "Roof & Wall Panel Installation",
        "description": "Roof panels, insulation, wall panels",
        "primary_agent": "roof_robot",
        "depends_on": 1,
    },
    3: {
        "name": "Electrical & Interior Finishing",
        "description": "Wiring, electrical components, interior",
        "primary_agent": "electrical_robot",
        "depends_on": 2,
    },
}

# =============================================================================
# METRICS COLLECTION
# =============================================================================
SCENARIOS = ["baseline", "nash_only", "nash_llm"]
SCENARIO_LABELS = {
    "baseline": "Baseline (FCFS)",
    "nash_only": "Nash Bargaining Only",
    "nash_llm": "Nash + LLM Negotiation",
}

# =============================================================================
# OUTPUT PATHS
# =============================================================================
RESULTS_DIR = "results"
GANTT_CHART_PATH = f"{RESULTS_DIR}/gantt_chart.png"
COMPLETION_CHART_PATH = f"{RESULTS_DIR}/completion_comparison.png"
RECOVERY_CHART_PATH = f"{RESULTS_DIR}/disruption_recovery.png"
NEGOTIATION_LOG_PATH = f"{RESULTS_DIR}/negotiation_log.txt"
