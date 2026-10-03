"""
EXPERT DATA COLLECTION POLICY v12 — smooth expert + CALLBACK-SIDE EXPLORATION
============================================================================

Deadband-override smooth expert with three exploration systems added INSIDE the
controller (so noisy actions reach EnergyPlus and the resulting next-state /
reward correspond to the executed action):

  expert action -> exploration -> actuation -> logging

  1) Safe local epsilon-greedy GLAZING (doc: "Safe epsilon-greedy glazing"):
     - applied only when glazing is scheduled to update (hourly at 6 steps/h);
     - epsilon on the JOINT decision, randomizing one (usually) or two windows;
     - respects one-tier-at-a-time and the glare rule (never lightens in glare).

  2) Dimming-space LIGHTING noise (doc: "Lighting exploration in dimming space"):
     - per-zone Gaussian noise in dimming fraction (sigma * nominal power);
     - only while occupied and on a lighting-update step;
     - bounded so artificial light keeps total < LIGHTING_SAFE_MAX_LUX.

  3) Per-zone stateful mixed THERMAL policy (doc: "Mixed thermal controller"):
     - outside 21-24: recovery (22/22) or occasionally noisy recovery;
     - inside 21-24: idle (16/28) / random-inactive band / active target;
     - each zone holds its chosen behavior for THERMAL_POLICY_HOLD_STEPS;
     - leaving the band immediately discards the held exploratory behavior.

policy_memory is updated with the EXECUTED (noisy) actions. The SDAR mask/prev/
mix use the executed action (as before). A separate `exploration_data` log keeps
the expert (clean) action, executed action, and exploration masks.

Action vector layout (all normalized to [-1, 1]):
  [0:4] glazing tier | [4:9] lighting | [9:14] heating SP | [14:19] cooling SP
"""

import numpy as np


# ============================================================
# PICK YOUR EXPERT
# ============================================================

EXPERT_MODE = "smooth"

EXPERT_ID_MAP = {
    "balanced": 0, "energy": 1, "comfort": 2, "visual": 3, "smooth": 4,
    "fast_responsive": 5, "slow_smooth": 6, "staggered_multirate": 7,
    "bad_energy_lazy": 8, "adaptive_balanced_good": 9,
}

# ============================================================
# THERMAL-TRIM DEFAULTS
# ============================================================
THERMAL_DEFAULTS = {
    "opt_start_window_hrs":  2.0,
    "opt_start_gain":        0.5,
    "solar_ref":             600.0,
    "solar_heat_offset_max": 1.0,
    "solar_cool_offset_max": 0.8,
    "cool_opt_gain":         0.5,
    "cool_drift_gain":       0.5,
}


# ============================================================
# PROFILES
# ============================================================
EXPERT_PROFILES = {
    "smooth": {
        "occ_heat_sp": 21.0,   "occ_cool_sp": 24.0,
        "pre_heat_sp": 20.0,   "pre_cool_sp": 25.0,
        "sb_heat_sp":  16.0,   "sb_cool_sp":  28.0,
        "target_lux": 500.0,
        "lighting_deadband_lux": 50.0,
        "lighting_action_deadband_w": 50.0,
        "solar_cool_darken":  (200.0, 350.0, 500.0),
        "solar_cool_lighten":  (80.0, 230.0, 380.0),
        "solar_heat_darken":  (400.0, 600.0, 800.0),
        "solar_heat_lighten": (270.0, 470.0, 670.0),
        "solar_neutral_darken":  (300.0, 500.0, 700.0),
        "solar_neutral_lighten": (180.0, 380.0, 580.0),
        "wpi_glare_hard": 1000.0,
        "glare_state": "sr2_ec03",
        "action_hold_steps": 6,
        "glazing_update_steps": 6,
        "lighting_update_steps": 1,
        "thermal_update_steps": 1,
        "one_tier_at_a_time": True,
        "opt_start_window_hrs":  2.0,
        "opt_start_gain":        0.35,
        "solar_ref":             700.0,
        "solar_heat_offset_max": 0.8,
        "solar_cool_offset_max": 0.6,
        "cool_opt_gain":         0.35,
        "cool_drift_gain":       0.4,
    },
}


# ============================================================
# APPLY SELECTED PROFILE
# ============================================================

if EXPERT_MODE not in EXPERT_PROFILES:
    raise ValueError(f"Unknown EXPERT_MODE: {EXPERT_MODE!r}. "
                     f"Choose from {list(EXPERT_PROFILES.keys())}")

PROFILE = EXPERT_PROFILES[EXPERT_MODE]
EXPERT_ID = EXPERT_ID_MAP[EXPERT_MODE]
print(f"[v12 controller] EXPERT_MODE = '{EXPERT_MODE}'  (id={EXPERT_ID})")


# ============================================================
# CONFIGURATION
# ============================================================

ZONES_MID = [
    'Perimeter_mid_ZN_1', 'Perimeter_mid_ZN_2', 'Perimeter_mid_ZN_3',
    'Perimeter_mid_ZN_4', 'Core_mid',
]

WINDOW_ZONES_MID = {
    'Perimeter_mid_ZN_1': 'Perimeter_mid_ZN_1_Wall_South_Window',
    'Perimeter_mid_ZN_2': 'Perimeter_mid_ZN_2_Wall_East_Window',
    'Perimeter_mid_ZN_3': 'Perimeter_mid_ZN_3_Wall_North_Window',
    'Perimeter_mid_ZN_4': 'Perimeter_mid_ZN_4_Wall_West_Window',
}

AVAILABLE_GLAZING_STATES = ['sr2_ec01', 'sr2_ec02', 'sr2_ec03', 'sr2_ec04']
GLAZING_CLEAR_TO_DARK   = ['sr2_ec01', 'sr2_ec02', 'sr2_ec03', 'sr2_ec04']
TIER_OF = {name: i for i, name in enumerate(AVAILABLE_GLAZING_STATES)}

LIGHTING_POWER_RANGE = (0.0, 2500.0)
HEATING_SETPOINT_RANGE = (16.0, 24.0)
COOLING_SETPOINT_RANGE = (22.0, 28.0)
DEADBAND = 1.0

# ---- deadband (comfort-band) override thermostat ----
COMFORT_BAND_LOW    = 22.0
COMFORT_BAND_HIGH   = 25.0
COMFORT_OVERRIDE_SP = 24.0

LIGHTING_TO_LUX = 0.4

LIGHTING_INSTALLED_W = {
    'Perimeter_mid_ZN_1': 2231.0,
    'Perimeter_mid_ZN_2': 2231.0,
    'Perimeter_mid_ZN_3': 1412.0,
    'Perimeter_mid_ZN_4': 1412.0,
    'Core_mid':          10586.0,
}

TARGET_LUX                  = PROFILE["target_lux"]
LIGHTING_DEADBAND_LUX       = PROFILE["lighting_deadband_lux"]
LIGHTING_ACTION_DEADBAND_W  = PROFILE["lighting_action_deadband_w"]
CORE_OCC_POWER              = TARGET_LUX / LIGHTING_TO_LUX

OCCUPIED_START = 8
OCCUPIED_END = 16
PRECONDITION_LEAD_HRS = 1

OCCUPIED_HEAT_SP     = PROFILE["occ_heat_sp"]
OCCUPIED_COOL_SP     = PROFILE["occ_cool_sp"]
PRECONDITION_HEAT_SP = PROFILE["pre_heat_sp"]
PRECONDITION_COOL_SP = PROFILE["pre_cool_sp"]
SETBACK_HEAT_SP      = PROFILE["sb_heat_sp"]
SETBACK_COOL_SP      = PROFILE["sb_cool_sp"]

SOLAR_COOL_DARKEN     = PROFILE["solar_cool_darken"]
SOLAR_COOL_LIGHTEN    = PROFILE["solar_cool_lighten"]
SOLAR_HEAT_DARKEN     = PROFILE["solar_heat_darken"]
SOLAR_HEAT_LIGHTEN    = PROFILE["solar_heat_lighten"]
SOLAR_NEUTRAL_DARKEN  = PROFILE["solar_neutral_darken"]
SOLAR_NEUTRAL_LIGHTEN = PROFILE["solar_neutral_lighten"]

WPI_GLARE_HARD     = PROFILE["wpi_glare_hard"]
GLARE_STATE        = PROFILE["glare_state"]
ACTION_HOLD_STEPS  = PROFILE["action_hold_steps"]
GLAZING_UPDATE_STEPS = PROFILE.get("glazing_update_steps", PROFILE["action_hold_steps"])
LIGHTING_UPDATE_STEPS = PROFILE.get("lighting_update_steps", 1)
THERMAL_UPDATE_STEPS = PROFILE.get("thermal_update_steps", 1)
ONE_TIER_AT_A_TIME = PROFILE["one_tier_at_a_time"]

OPT_START_WINDOW_HRS  = PROFILE.get("opt_start_window_hrs",  THERMAL_DEFAULTS["opt_start_window_hrs"])
OPT_START_GAIN        = PROFILE.get("opt_start_gain",        THERMAL_DEFAULTS["opt_start_gain"])
SOLAR_REF             = PROFILE.get("solar_ref",             THERMAL_DEFAULTS["solar_ref"])
SOLAR_HEAT_OFFSET_MAX = PROFILE.get("solar_heat_offset_max", THERMAL_DEFAULTS["solar_heat_offset_max"])
SOLAR_COOL_OFFSET_MAX = PROFILE.get("solar_cool_offset_max", THERMAL_DEFAULTS["solar_cool_offset_max"])
COOL_OPT_GAIN         = PROFILE.get("cool_opt_gain",         THERMAL_DEFAULTS["cool_opt_gain"])
COOL_DRIFT_GAIN       = PROFILE.get("cool_drift_gain",       THERMAL_DEFAULTS["cool_drift_gain"])

COIL_ACTIVE_EPS = 10.0


# ============================================================
# SDAR ACTION VECTOR LAYOUT
# ============================================================
ACTION_DIM   = 19
GLAZE_SLICE  = slice(0, 4)
LIGHT_SLICE  = slice(4, 9)
HEAT_SLICE   = slice(9, 14)
COOL_SLICE   = slice(14, 19)

GLAZE_RANGE  = (0.0, 3.0)
LIGHT_RANGE  = (0.0, 2500.0)
HEAT_RANGE   = (16.0, 24.0)
COOL_RANGE   = (22.0, 28.0)

WINDOW_ORDER = list(WINDOW_ZONES_MID.keys())
ZONE_ORDER   = list(ZONES_MID)

LIGHT_RANGE_PER_ZONE = {zone: (0.0, LIGHTING_INSTALLED_W[zone]) for zone in ZONE_ORDER}

ACTION_MASK_VALUE = -2.0
MASK_EPS = 1e-3

ACTION_LAYOUT = {
    "glazing":  (GLAZE_SLICE, WINDOW_ORDER),
    "lighting": (LIGHT_SLICE, ZONE_ORDER),
    "heating":  (HEAT_SLICE,  ZONE_ORDER),
    "cooling":  (COOL_SLICE,  ZONE_ORDER),
    "ranges":   {"glaze": GLAZE_RANGE, "light": LIGHT_RANGE_PER_ZONE,
                 "heat":  HEAT_RANGE,  "cool":  COOL_RANGE},
    "mask_value": ACTION_MASK_VALUE,
    "dim": ACTION_DIM,
}


# ============================================================
# EXPLORATION CONFIG
# ============================================================

EPISODE_ID = 13                 # change for every rollout
NOISE_PROFILE = "high"       # "clean", "low", "medium", "high"

EXPLORATION_ENABLED = NOISE_PROFILE != "clean"

NOISE_PROFILES = {
    "clean": {
        "glazing_epsilon": 0.00,
        "lighting_noise_prob": 0.00,
        "lighting_noise_std_frac": 0.00,
        "inside_idle_prob": 1.00,
        "inside_random_band_prob": 0.00,
        "inside_active_target_prob": 0.00,
        "outside_noisy_target_prob": 0.00,
        "outside_target_noise_std_c": 0.00,
    },
    "low": {
        "glazing_epsilon": 0.05,
        "lighting_noise_prob": 0.15,
        "lighting_noise_std_frac": 0.03,
        "inside_idle_prob": 0.85,
        "inside_random_band_prob": 0.10,
        "inside_active_target_prob": 0.05,
        "outside_noisy_target_prob": 0.10,
        "outside_target_noise_std_c": 0.25,
    },
    "medium": {
        "glazing_epsilon": 0.10,
        "lighting_noise_prob": 0.30,
        "lighting_noise_std_frac": 0.05,
        "inside_idle_prob": 0.70,
        "inside_random_band_prob": 0.20,
        "inside_active_target_prob": 0.10,
        "outside_noisy_target_prob": 0.20,
        "outside_target_noise_std_c": 0.50,
    },
    "high": {
        "glazing_epsilon": 0.20,
        "lighting_noise_prob": 0.40,
        "lighting_noise_std_frac": 0.08,
        "inside_idle_prob": 0.55,
        "inside_random_band_prob": 0.30,
        "inside_active_target_prob": 0.15,
        "outside_noisy_target_prob": 0.30,
        "outside_target_noise_std_c": 0.70,
    },
}

P = NOISE_PROFILES[NOISE_PROFILE]

GLAZING_EPSILON = P["glazing_epsilon"]

LIGHTING_NOISE_PROB = P["lighting_noise_prob"]
LIGHTING_NOISE_STD_FRAC = P["lighting_noise_std_frac"]
LIGHTING_SAFE_MAX_LUX = 1000.0

INSIDE_IDLE_PROB = P["inside_idle_prob"]
INSIDE_RANDOM_BAND_PROB = P["inside_random_band_prob"]
INSIDE_ACTIVE_TARGET_PROB = P["inside_active_target_prob"]

OUTSIDE_NOISY_TARGET_PROB = P["outside_noisy_target_prob"]
OUTSIDE_TARGET_NOISE_STD_C = P["outside_target_noise_std_c"]

THERMAL_POLICY_HOLD_STEPS = 6

# SDAR temporal exploration: probability of repeating the previous action
SDAR_GLAZING_REPEAT_PROB = .10
SDAR_LIGHTING_REPEAT_PROB = .10
SDAR_THERMAL_REPEAT_PROB = .05

THERMAL_TARGET_SP = 22.0
# Keep THERMAL_RANDOM_TARGET_LOW = 22.0: the cooling action range begins at
# 22 C, so a 21 C target would normalize the cooling action below -1.
THERMAL_RANDOM_TARGET_LOW = 22.0
THERMAL_RANDOM_TARGET_HIGH = 24.0
THERMAL_INACTIVE_MARGIN_C = 0.25

# Unique deterministic seed for every expert and episode
EXPLORATION_SEED = (
    10_000
    + 1_000 * EXPERT_ID
    + EPISODE_ID+ 42142142142
)

rng = np.random.default_rng(EXPLORATION_SEED)

# Validate thermal probabilities
inside_probability_sum = (
    INSIDE_IDLE_PROB
    + INSIDE_RANDOM_BAND_PROB
    + INSIDE_ACTIVE_TARGET_PROB
)

if not np.isclose(inside_probability_sum, 1.0):
    raise ValueError(
        "Thermal inside-band probabilities must sum to 1. "
        f"Current sum: {inside_probability_sum}"
    )


# ============================================================
# DATA STORAGE
# ============================================================

lighting_data = {zone: [] for zone in ZONES_MID}
temperature_data = {zone: [] for zone in ZONES_MID}
solar_data = {zone: [] for zone in WINDOW_ZONES_MID}
ext_irr_data = {zone: [] for zone in WINDOW_ZONES_MID}
wpi_data = {zone: [] for zone in WINDOW_ZONES_MID}

glazing_state_data = {zone: [] for zone in WINDOW_ZONES_MID}
lighting_power_data = {zone: [] for zone in ZONES_MID}
heating_setpoint_data = {zone: [] for zone in ZONES_MID}
cooling_setpoint_data = {zone: [] for zone in ZONES_MID}

meter_data = {
    'Heating:Electricity': [],
    'Cooling:Electricity': [],
    'InteriorLights:Electricity': [],
    'Electricity:HVAC': [],
}

temporal_data = {
    'hour': [], 'minute': [], 'day_of_year': [], 'weekday': [],
    'hour_sin': [], 'hour_cos': [], 'doy_sin': [], 'doy_cos': [],
    'is_occupied': [], 'is_preconditioning': [],
    'expert_id': [],
}

# SDAR per-step logs (action = EXECUTED / noisy action)
action_vector_data = {
    "action":         [],
    "prev_action":    [],
    "selection_mask": [],
    "action_mix":     [],
}

# Exploration log: expert (clean) vs executed (noisy) + which dims explored.
exploration_data = {
    "expert_action": [],
    "executed_action": [],
    "exploration_mask": [],
    "glazing_explored": [],
    "lighting_explored": [],
    "thermal_mode": [],
    "epsilon": [],
    "sdar_forced_repeat": [],
    "seed": EXPLORATION_SEED,
    "episode_id": EPISODE_ID,
    "noise_profile": NOISE_PROFILE,
}

output_variable_data = {
    'HVAC_electricity_demand_rate': [],
}


# ============================================================
# POLICY MEMORY
# ============================================================

policy_memory = {
    "expert_mode": EXPERT_MODE,
    "expert_id": EXPERT_ID,
    "step": 0,
    "glazing": {zone: "sr2_ec01" for zone in WINDOW_ZONES_MID},
    "lighting": {zone: 0.0 for zone in ZONES_MID},
    "heating": {zone: SETBACK_HEAT_SP for zone in ZONES_MID},
    "cooling": {zone: SETBACK_COOL_SP for zone in ZONES_MID},
    "cooling_active": False,
    "heating_active": False,
    "prev_action_vec": None,
    # per-zone thermal exploration state (held behavior)
    "thermal_exploration": {
        zone: {
            "heating_sp": HEATING_SETPOINT_RANGE[0],
            "cooling_sp": COOLING_SETPOINT_RANGE[1],
            "mode": "idle",
            "remaining": 0,
        }
        for zone in ZONES_MID
    },
}

meter_handles = {
    "Heating:Electricity": None,
    "Cooling:Electricity": None,
    "InteriorLights:Electricity": None,
    "Electricity:HVAC": None
}


# ============================================================
# HELPERS
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def get_meter_value_cached(state, meter_name):
    if meter_handles[meter_name] is None:
        meter_handles[meter_name] = eps.api.exchange.get_meter_handle(state, meter_name)
    return eps.api.exchange.get_meter_value(state, meter_handles[meter_name])


def is_occupied(dt):
    return OCCUPIED_START <= dt.hour < OCCUPIED_END


def is_preconditioning(dt):
    return (OCCUPIED_START - PRECONDITION_LEAD_HRS) <= dt.hour < OCCUPIED_START


def compute_temporal_features(dt):
    hour = dt.hour
    minute = dt.minute
    doy = dt.timetuple().tm_yday
    weekday = dt.weekday()

    tod_frac = (hour + minute / 60.0) / 24.0
    hour_sin = float(np.sin(2.0 * np.pi * tod_frac))
    hour_cos = float(np.cos(2.0 * np.pi * tod_frac))

    doy_frac = (doy - 1) / 365.0
    doy_sin = float(np.sin(2.0 * np.pi * doy_frac))
    doy_cos = float(np.cos(2.0 * np.pi * doy_frac))

    return {
        'hour': hour, 'minute': minute, 'day_of_year': doy, 'weekday': weekday,
        'hour_sin': hour_sin, 'hour_cos': hour_cos,
        'doy_sin': doy_sin, 'doy_cos': doy_cos,
        'is_occupied': bool(is_occupied(dt)),
        'is_preconditioning': bool(is_preconditioning(dt)),
    }


def _norm(x, lo, hi):
    """Affine map from [lo, hi] -> [-1, 1]."""
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def encode_action_vec(glaze, light, heat, cool):
    """Pack expert per-zone actions into a 19-dim normalized vector in [-1, 1]."""
    v = np.zeros(ACTION_DIM, dtype=np.float32)
    for i, zone in enumerate(WINDOW_ORDER):
        v[i] = _norm(float(TIER_OF[glaze[zone]]), *GLAZE_RANGE)
    for i, zone in enumerate(ZONE_ORDER):
        v[4 + i]  = _norm(light[zone], *LIGHT_RANGE_PER_ZONE[zone])
        v[9 + i]  = _norm(heat[zone],  *HEAT_RANGE)
        v[14 + i] = _norm(cool[zone],  *COOL_RANGE)
    return v


# ============================================================
# EXPERT RULES
# ============================================================

def choose_glazing_expert(occupied, cooling_active, heating_active,
                          ext_irr, wpi, current_glazing):
    if not occupied:
        return 'sr2_ec01'

    if wpi >= WPI_GLARE_HARD:
        current_tier = TIER_OF.get(current_glazing, 0)
        glare_tier = TIER_OF[GLARE_STATE]
        return AVAILABLE_GLAZING_STATES[max(current_tier, glare_tier)]

    if cooling_active:
        darken, lighten = SOLAR_COOL_DARKEN, SOLAR_COOL_LIGHTEN
    elif heating_active:
        darken, lighten = SOLAR_HEAT_DARKEN, SOLAR_HEAT_LIGHTEN
    else:
        darken, lighten = SOLAR_NEUTRAL_DARKEN, SOLAR_NEUTRAL_LIGHTEN

    current_tier = TIER_OF.get(current_glazing, 0)

    target = current_tier
    while target < 3 and ext_irr > darken[target]:
        target += 1
    if target > current_tier:
        if ONE_TIER_AT_A_TIME:
            return AVAILABLE_GLAZING_STATES[current_tier + 1]
        return AVAILABLE_GLAZING_STATES[target]

    target = current_tier
    while target > 0 and ext_irr < lighten[target - 1]:
        target -= 1
    if target < current_tier:
        if ONE_TIER_AT_A_TIME:
            return AVAILABLE_GLAZING_STATES[current_tier - 1]
        return AVAILABLE_GLAZING_STATES[target]

    return current_glazing


def choose_perimeter_lighting_expert(occupied, current_wpi, zone_max_w,
                                     prev_power=0.0,
                                     action_deadband_w=0.0):
    if not occupied:
        return 0.0
    shortfall_lux = TARGET_LUX - current_wpi
    if shortfall_lux < LIGHTING_DEADBAND_LUX:
        return 0.0
    desired_power = clamp(shortfall_lux / LIGHTING_TO_LUX, 0.0, zone_max_w)
    if action_deadband_w > 0.0 \
       and abs(desired_power - prev_power) < action_deadband_w:
        return prev_power
    return desired_power


def choose_core_lighting_expert(occupied):
    if not occupied:
        return 0.0
    return CORE_OCC_POWER


def choose_hvac_zone(dt, zone_temp, zone_ext_irr,
                     heating_active=False, cooling_active=False,
                     has_window=True):
    """
    DEADBAND (comfort-band) override thermostat (deterministic expert).
      - INSIDE [21,24] -> 16/28 (no coil fires).
      - OUTSIDE       -> 22/22 (recover toward 22).
    """
    if COMFORT_BAND_LOW <= zone_temp <= COMFORT_BAND_HIGH:
        h = HEATING_SETPOINT_RANGE[0]
        c = COOLING_SETPOINT_RANGE[1]
    else:
        h = COMFORT_OVERRIDE_SP
        c = COMFORT_OVERRIDE_SP
    return h, c


# ============================================================
# EXPLORATION — glazing (safe local epsilon-greedy)
# ============================================================

def apply_glazing_exploration(selected_g, current_obs, update_glazing):
    """
    With prob GLAZING_EPSILON (on glazing-update steps only), modify one or two
    glazing zones. Respects the update period, one-tier-at-a-time transitions,
    and the glare rule (never lightens during glare).
    """
    explored = {zone: False for zone in WINDOW_ORDER}

    if (not EXPLORATION_ENABLED
            or not update_glazing
            or rng.random() >= GLAZING_EPSILON):
        return selected_g, explored

    n_zones = 1 if rng.random() < 0.8 else 2
    zones_to_explore = rng.choice(WINDOW_ORDER, size=n_zones, replace=False)

    result = selected_g.copy()
    for zone in zones_to_explore:
        current_tier = TIER_OF[policy_memory["glazing"][zone]]
        expert_tier = TIER_OF[selected_g[zone]]

        candidate_tiers = [
            tier for tier in (current_tier - 1, current_tier, current_tier + 1)
            if 0 <= tier <= 3
        ]

        if current_obs["wpi"][zone] >= WPI_GLARE_HARD:
            minimum_safe_tier = max(current_tier, TIER_OF[GLARE_STATE])
            candidate_tiers = [t for t in candidate_tiers if t >= minimum_safe_tier]

        alternatives = [t for t in candidate_tiers if t != expert_tier]
        if alternatives:
            chosen_tier = int(rng.choice(alternatives))
            result[zone] = AVAILABLE_GLAZING_STATES[chosen_tier]
            explored[zone] = True

    return result, explored


# ============================================================
# EXPLORATION — lighting (bounded Gaussian in dimming space)
# ============================================================

def apply_lighting_exploration(selected_l, current_obs, occupied, update_lighting):
    """Add bounded Gaussian noise in dimming-fraction space (occupied only)."""
    explored = {zone: False for zone in ZONE_ORDER}

    if (not EXPLORATION_ENABLED or not update_lighting or not occupied):
        return selected_l, explored

    result = selected_l.copy()
    for zone in ZONE_ORDER:
        if rng.random() >= LIGHTING_NOISE_PROB:
            continue

        nominal_w = LIGHTING_INSTALLED_W[zone]
        expert_dimming = np.clip(selected_l[zone] / nominal_w, 0.0, 1.0)
        noisy_dimming = np.clip(
            expert_dimming + rng.normal(0.0, LIGHTING_NOISE_STD_FRAC), 0.0, 1.0)
        noisy_power_w = noisy_dimming * nominal_w

        maximum_power_w = min(nominal_w, LIGHTING_POWER_RANGE[1])
        if zone in WINDOW_ZONES_MID:
            daylight_lux = current_obs["wpi"][zone]
            maximum_power_by_lux = max(
                0.0, (LIGHTING_SAFE_MAX_LUX - daylight_lux) / LIGHTING_TO_LUX)
            maximum_power_w = min(maximum_power_w, maximum_power_by_lux)

        result[zone] = clamp(noisy_power_w, 0.0, maximum_power_w)
        explored[zone] = not np.isclose(result[zone], selected_l[zone])

    return result, explored


# ============================================================
# EXPLORATION — per-zone mixed thermal policy
# ============================================================

def choose_exploratory_hvac_zone(zone, zone_temp):
    """
    Per-zone stateful mixed thermal behavior.
      OUTSIDE 21-24 : recovery (22/22) or occasional noisy recovery.
      INSIDE  21-24 : idle (16/28) / random-inactive band / active target,
                      held for THERMAL_POLICY_HOLD_STEPS.
    Leaving the band immediately discards any held exploratory behavior.
    """
    memory = policy_memory["thermal_exploration"][zone]
    inside_comfort = COMFORT_BAND_LOW <= zone_temp <= COMFORT_BAND_HIGH

    # ---- outside comfort: recovery takes precedence ----
    if not inside_comfort:
        memory["remaining"] = 0
        if EXPLORATION_ENABLED and rng.random() < OUTSIDE_NOISY_TARGET_PROB:
            target = THERMAL_TARGET_SP + rng.normal(0.0, OUTSIDE_TARGET_NOISE_STD_C)
            target = clamp(target, THERMAL_RANDOM_TARGET_LOW, THERMAL_RANDOM_TARGET_HIGH)
            mode = "noisy_recovery"
        else:
            target = THERMAL_TARGET_SP
            mode = "expert_recovery"
        memory["heating_sp"] = target
        memory["cooling_sp"] = target
        memory["mode"] = mode
        return target, target, mode

    # ---- inside comfort: hold prior exploratory behavior ----
    if memory["remaining"] > 0:
        memory["remaining"] -= 1
        return memory["heating_sp"], memory["cooling_sp"], memory["mode"]

    # exploration disabled -> plain idle expert inside the band
    if not EXPLORATION_ENABLED:
        memory["heating_sp"] = HEATING_SETPOINT_RANGE[0]
        memory["cooling_sp"] = COOLING_SETPOINT_RANGE[1]
        memory["mode"] = "idle"
        memory["remaining"] = 0
        return HEATING_SETPOINT_RANGE[0], COOLING_SETPOINT_RANGE[1], "idle"

    # select a new inside-band behavior
    sample = rng.random()
    if sample < INSIDE_IDLE_PROB:
        heating_sp = HEATING_SETPOINT_RANGE[0]   # 16
        cooling_sp = COOLING_SETPOINT_RANGE[1]   # 28
        mode = "idle"
    elif sample < (INSIDE_IDLE_PROB + INSIDE_RANDOM_BAND_PROB):
        maximum_heating_sp = min(zone_temp - THERMAL_INACTIVE_MARGIN_C,
                                 HEATING_SETPOINT_RANGE[1])
        minimum_cooling_sp = max(zone_temp + THERMAL_INACTIVE_MARGIN_C,
                                 COOLING_SETPOINT_RANGE[0])
        heating_sp = rng.uniform(HEATING_SETPOINT_RANGE[0], maximum_heating_sp)
        cooling_sp = rng.uniform(minimum_cooling_sp, COOLING_SETPOINT_RANGE[1])
        mode = "random_inactive_band"
    else:
        target = rng.uniform(THERMAL_RANDOM_TARGET_LOW, THERMAL_RANDOM_TARGET_HIGH)
        heating_sp = target
        cooling_sp = target
        mode = "active_exploration"

    memory["heating_sp"] = float(heating_sp)
    memory["cooling_sp"] = float(cooling_sp)
    memory["mode"] = mode
    memory["remaining"] = THERMAL_POLICY_HOLD_STEPS - 1
    return float(heating_sp), float(cooling_sp), mode


def apply_sdar_repeat_exploration(
    selected_g,
    selected_l,
    selected_h,
    selected_c,
    current_obs,
    thermal_mode,
    thermal_mem_snapshot,
):
    """
    Occasionally repeat the previous executed action.

    This creates genuine SDAR b=0 samples. Current action-value noise creates
    b=1 samples, so together they improve SDAR update/repeat coverage.

    When a THERMAL repeat fires, the exploratory thermal call has already mutated
    policy_memory["thermal_exploration"][zone] (remaining/mode/held setpoints) and
    set thermal_mode[zone] to the NEW, now-unexecuted policy. So on a thermal
    repeat we also restore that zone's thermal memory to its pre-call snapshot and
    report thermal_mode = "sdar_repeat", keeping mode/memory consistent with what
    actually executed.
    """
    forced_repeat_mask = np.zeros(ACTION_DIM, dtype=np.float32)

    # No previous action exists on the first timestep
    if not EXPLORATION_ENABLED or policy_memory["prev_action_vec"] is None:
        return (
            selected_g,
            selected_l,
            selected_h,
            selected_c,
            forced_repeat_mask,
        )

    # Glazing: independent decision for each perimeter zone
    for i, zone in enumerate(WINDOW_ORDER):
        if rng.random() < SDAR_GLAZING_REPEAT_PROB:
            selected_g[zone] = policy_memory["glazing"][zone]
            forced_repeat_mask[i] = 1.0

    # Lighting: independent decision for all five zones
    for i, zone in enumerate(ZONE_ORDER):
        if rng.random() < SDAR_LIGHTING_REPEAT_PROB:
            selected_l[zone] = policy_memory["lighting"][zone]
            forced_repeat_mask[4 + i] = 1.0

    # Thermal: repeat heating and cooling jointly for each zone.
    # Only allow this inside the comfort band; recovery is never blocked.
    for i, zone in enumerate(ZONE_ORDER):
        zone_temp = current_obs["temp"][zone]
        inside_comfort = COMFORT_BAND_LOW <= zone_temp <= COMFORT_BAND_HIGH

        if inside_comfort and rng.random() < SDAR_THERMAL_REPEAT_PROB:
            selected_h[zone] = policy_memory["heating"][zone]
            selected_c[zone] = policy_memory["cooling"][zone]

            forced_repeat_mask[9 + i] = 1.0
            forced_repeat_mask[14 + i] = 1.0

            # restore the thermal memory that the exploratory call mutated, and
            # report the repeat as its own mode (consistent with what executed).
            policy_memory["thermal_exploration"][zone] = dict(
                thermal_mem_snapshot[zone])
            thermal_mode[zone] = "sdar_repeat"

    return (
        selected_g,
        selected_l,
        selected_h,
        selected_c,
        forced_repeat_mask,
    )


# ============================================================
# DECISION DISPATCH — expert action -> exploration -> memory update
# ============================================================

def choose_expert_actions(current_obs, dt):
    policy_memory["step"] += 1
    update_glazing  = (policy_memory["step"] % GLAZING_UPDATE_STEPS == 0)
    update_lighting = (policy_memory["step"] % LIGHTING_UPDATE_STEPS == 0)
    update_thermal  = (policy_memory["step"] % THERMAL_UPDATE_STEPS == 0)

    occupied = is_occupied(dt)
    cooling_active = policy_memory["cooling_active"]
    heating_active = policy_memory["heating_active"]

    selected_g, selected_l, selected_h, selected_c = {}, {}, {}, {}
    base_h, base_c = {}, {}   # deterministic thermal expert (for exploration mask)
    thermal_mode = {}
    thermal_mem_snapshot = {}  # pre-call thermal_exploration memory (for SDAR repeat restore)

    for zone, _window in WINDOW_ZONES_MID.items():
        wpi = current_obs["wpi"][zone]
        ext_irr = current_obs["ext_irr"][zone]
        current_g = policy_memory["glazing"][zone]

        if update_glazing:
            g = choose_glazing_expert(
                occupied=occupied,
                cooling_active=cooling_active,
                heating_active=heating_active,
                ext_irr=ext_irr,
                wpi=wpi,
                current_glazing=current_g,
            )
        else:
            g = current_g
        selected_g[zone] = g

        if update_lighting:
            target_p = choose_perimeter_lighting_expert(
                occupied=occupied,
                current_wpi=wpi,
                zone_max_w=LIGHTING_INSTALLED_W[zone],
                prev_power=policy_memory["lighting"][zone],
                action_deadband_w=LIGHTING_ACTION_DEADBAND_W,
            )
            selected_l[zone] = clamp(target_p, *LIGHTING_POWER_RANGE)
        else:
            selected_l[zone] = policy_memory["lighting"][zone]

        # Deterministic thermal expert (logging baseline for the mask)
        h_expert, c_expert = choose_hvac_zone(
            dt=dt,
            zone_temp=current_obs["temp"][zone],
            zone_ext_irr=current_obs["ext_irr"][zone],
            heating_active=heating_active,
            cooling_active=cooling_active,
            has_window=True,
        )
        base_h[zone] = h_expert
        base_c[zone] = c_expert

        # Thermal — per-zone mixed exploratory policy (actually executed).
        # Snapshot the pre-call thermal memory so a later SDAR forced-repeat can
        # restore it (choose_exploratory_hvac_zone mutates it in place).
        thermal_mem_snapshot[zone] = dict(policy_memory["thermal_exploration"][zone])
        if update_thermal:
            h_z, c_z, m_z = choose_exploratory_hvac_zone(
                zone=zone, zone_temp=current_obs["temp"][zone])
            selected_h[zone] = h_z
            selected_c[zone] = c_z
            thermal_mode[zone] = m_z
        else:
            selected_h[zone] = policy_memory["heating"][zone]
            selected_c[zone] = policy_memory["cooling"][zone]
            thermal_mode[zone] = policy_memory["thermal_exploration"][zone]["mode"]

    core_zone = "Core_mid"

    if update_lighting:
        selected_l[core_zone] = clamp(choose_core_lighting_expert(occupied),
                                      *LIGHTING_POWER_RANGE)
    else:
        selected_l[core_zone] = policy_memory["lighting"][core_zone]

    # Deterministic thermal expert for the core (logging baseline)
    h_expert_c, c_expert_c = choose_hvac_zone(
        dt=dt,
        zone_temp=current_obs["temp"][core_zone],
        zone_ext_irr=0.0,
        heating_active=heating_active,
        cooling_active=cooling_active,
        has_window=False,
    )
    base_h[core_zone] = h_expert_c
    base_c[core_zone] = c_expert_c

    thermal_mem_snapshot[core_zone] = dict(policy_memory["thermal_exploration"][core_zone])
    if update_thermal:
        h_c, c_c, m_c = choose_exploratory_hvac_zone(
            zone=core_zone, zone_temp=current_obs["temp"][core_zone])
        selected_h[core_zone] = h_c
        selected_c[core_zone] = c_c
        thermal_mode[core_zone] = m_c
    else:
        selected_h[core_zone] = policy_memory["heating"][core_zone]
        selected_c[core_zone] = policy_memory["cooling"][core_zone]
        thermal_mode[core_zone] = policy_memory["thermal_exploration"][core_zone]["mode"]

    # ---- preserve deterministic expert decision ----
    # glazing/lighting: pre-exploration values (exploration applied just below);
    # thermal: the deterministic base_h/base_c (selected_h/c already hold the
    # EXECUTED mixed-policy setpoints, so using them would zero the thermal mask).
    expert_g = selected_g.copy()
    expert_l = selected_l.copy()
    expert_h = base_h.copy()
    expert_c = base_c.copy()

    # ---- apply glazing + lighting exploration ----
    selected_g, glazing_explored = apply_glazing_exploration(
        selected_g=selected_g, current_obs=current_obs, update_glazing=update_glazing)
    selected_l, lighting_explored = apply_lighting_exploration(
        selected_l=selected_l, current_obs=current_obs,
        occupied=occupied, update_lighting=update_lighting)

    # ---- SDAR temporal-repeat exploration (forces genuine b=0 samples) ----
    # Reads policy_memory (the PREVIOUS executed action) before it is updated
    # below, so a forced repeat copies last step's executed value.
    (
        selected_g,
        selected_l,
        selected_h,
        selected_c,
        sdar_forced_repeat,
    ) = apply_sdar_repeat_exploration(
        selected_g,
        selected_l,
        selected_h,
        selected_c,
        current_obs,
        thermal_mode,
        thermal_mem_snapshot,
    )

    # ---- update memory using EXECUTED (noisy) actions ----
    for zone in WINDOW_ZONES_MID:
        policy_memory["glazing"][zone] = selected_g[zone]
    for zone in ZONES_MID:
        policy_memory["lighting"][zone] = selected_l[zone]
        policy_memory["heating"][zone] = selected_h[zone]
        policy_memory["cooling"][zone] = selected_c[zone]

    # ---- exploration logging (expert vs executed) ----
    expert_vec = encode_action_vec(expert_g, expert_l, expert_h, expert_c)
    executed_vec = encode_action_vec(selected_g, selected_l, selected_h, selected_c)
    exploration_mask = (np.abs(executed_vec - expert_vec) > MASK_EPS).astype(np.float32)

    exploration_data["expert_action"].append(expert_vec)
    exploration_data["executed_action"].append(executed_vec)
    exploration_data["exploration_mask"].append(exploration_mask)
    exploration_data["glazing_explored"].append(glazing_explored)
    exploration_data["lighting_explored"].append(lighting_explored)
    exploration_data["thermal_mode"].append(dict(thermal_mode))
    exploration_data["epsilon"].append(GLAZING_EPSILON)
    exploration_data["sdar_forced_repeat"].append(sdar_forced_repeat.copy())

    return selected_g, selected_l, selected_h, selected_c


# ============================================================
# CALLBACK
# ============================================================

def callback_func(state):
    if not eps.api.exchange.api_data_fully_ready(state):
        return
    if eps.api.exchange.warmup_flag(state):
        return

    dt = eps.get_datetime()

    light_mid_1 = eps.get_variable_value(name="Lights Electricity Rate", key="Perimeter_mid_ZN_1")
    temp_mid_1 = eps.get_variable_value(name="Zone Mean Air Temperature", key="Perimeter_mid_ZN_1")
    solar_mid_1 = eps.get_variable_value(
        name="Surface Window Transmitted Solar Radiation Rate",
        key="Perimeter_mid_ZN_1_Wall_South_Window")
    ext_irr_mid_1 = eps.get_variable_value(
        name="Surface Outside Face Incident Solar Radiation Rate per Area",
        key="Perimeter_mid_ZN_1_Wall_South_Window")

    light_mid_2 = eps.get_variable_value(name="Lights Electricity Rate", key="Perimeter_mid_ZN_2")
    temp_mid_2 = eps.get_variable_value(name="Zone Mean Air Temperature", key="Perimeter_mid_ZN_2")
    solar_mid_2 = eps.get_variable_value(
        name="Surface Window Transmitted Solar Radiation Rate",
        key="Perimeter_mid_ZN_2_Wall_East_Window")
    ext_irr_mid_2 = eps.get_variable_value(
        name="Surface Outside Face Incident Solar Radiation Rate per Area",
        key="Perimeter_mid_ZN_2_Wall_East_Window")

    light_mid_3 = eps.get_variable_value(name="Lights Electricity Rate", key="Perimeter_mid_ZN_3")
    temp_mid_3 = eps.get_variable_value(name="Zone Mean Air Temperature", key="Perimeter_mid_ZN_3")
    solar_mid_3 = eps.get_variable_value(
        name="Surface Window Transmitted Solar Radiation Rate",
        key="Perimeter_mid_ZN_3_Wall_North_Window")
    ext_irr_mid_3 = eps.get_variable_value(
        name="Surface Outside Face Incident Solar Radiation Rate per Area",
        key="Perimeter_mid_ZN_3_Wall_North_Window")

    light_mid_4 = eps.get_variable_value(name="Lights Electricity Rate", key="Perimeter_mid_ZN_4")
    temp_mid_4 = eps.get_variable_value(name="Zone Mean Air Temperature", key="Perimeter_mid_ZN_4")
    solar_mid_4 = eps.get_variable_value(
        name="Surface Window Transmitted Solar Radiation Rate",
        key="Perimeter_mid_ZN_4_Wall_West_Window")
    ext_irr_mid_4 = eps.get_variable_value(
        name="Surface Outside Face Incident Solar Radiation Rate per Area",
        key="Perimeter_mid_ZN_4_Wall_West_Window")

    light_core_mid = eps.get_variable_value(name="Lights Electricity Rate", key="Core_mid")
    temp_core_mid = eps.get_variable_value(name="Zone Mean Air Temperature", key="Core_mid")
    hvac_electricity_demand_rate = eps.get_variable_value(
        name="Facility Total HVAC Electricity Demand Rate", key="Whole Building")

    current_wpi = {}
    for zone, window in WINDOW_ZONES_MID.items():
        g_now = policy_memory["glazing"][zone]
        current_wpi[zone] = float(
            eps.calculate_wpi(zone=zone, cfs_name={window: g_now}).mean()
        )

    heating_elec = get_meter_value_cached(state, "Heating:Electricity")
    cooling_elec = get_meter_value_cached(state, "Cooling:Electricity")
    lights_elec  = get_meter_value_cached(state, "InteriorLights:Electricity")
    hvac_elec    = get_meter_value_cached(state, "Electricity:HVAC")

    policy_memory["cooling_active"] = (cooling_elec > COIL_ACTIVE_EPS)
    policy_memory["heating_active"] = (heating_elec > COIL_ACTIVE_EPS)

    meter_data['Heating:Electricity'].append(heating_elec)
    meter_data['Cooling:Electricity'].append(cooling_elec)
    meter_data['InteriorLights:Electricity'].append(lights_elec)
    meter_data['Electricity:HVAC'].append(hvac_elec)
    output_variable_data['HVAC_electricity_demand_rate'].append(hvac_electricity_demand_rate)

    tf = compute_temporal_features(dt)
    for k, v in tf.items():
        temporal_data[k].append(v)
    temporal_data['expert_id'].append(EXPERT_ID)

    lighting_data['Perimeter_mid_ZN_1'].append(light_mid_1)
    lighting_data['Perimeter_mid_ZN_2'].append(light_mid_2)
    lighting_data['Perimeter_mid_ZN_3'].append(light_mid_3)
    lighting_data['Perimeter_mid_ZN_4'].append(light_mid_4)
    lighting_data['Core_mid'].append(light_core_mid)

    temperature_data['Perimeter_mid_ZN_1'].append(temp_mid_1)
    temperature_data['Perimeter_mid_ZN_2'].append(temp_mid_2)
    temperature_data['Perimeter_mid_ZN_3'].append(temp_mid_3)
    temperature_data['Perimeter_mid_ZN_4'].append(temp_mid_4)
    temperature_data['Core_mid'].append(temp_core_mid)

    solar_data['Perimeter_mid_ZN_1'].append(solar_mid_1)
    solar_data['Perimeter_mid_ZN_2'].append(solar_mid_2)
    solar_data['Perimeter_mid_ZN_3'].append(solar_mid_3)
    solar_data['Perimeter_mid_ZN_4'].append(solar_mid_4)

    ext_irr_data['Perimeter_mid_ZN_1'].append(ext_irr_mid_1)
    ext_irr_data['Perimeter_mid_ZN_2'].append(ext_irr_mid_2)
    ext_irr_data['Perimeter_mid_ZN_3'].append(ext_irr_mid_3)
    ext_irr_data['Perimeter_mid_ZN_4'].append(ext_irr_mid_4)

    wpi_data['Perimeter_mid_ZN_1'].append(current_wpi['Perimeter_mid_ZN_1'])
    wpi_data['Perimeter_mid_ZN_2'].append(current_wpi['Perimeter_mid_ZN_2'])
    wpi_data['Perimeter_mid_ZN_3'].append(current_wpi['Perimeter_mid_ZN_3'])
    wpi_data['Perimeter_mid_ZN_4'].append(current_wpi['Perimeter_mid_ZN_4'])

    current_obs = {
        "temp": {
            'Perimeter_mid_ZN_1': temp_mid_1, 'Perimeter_mid_ZN_2': temp_mid_2,
            'Perimeter_mid_ZN_3': temp_mid_3, 'Perimeter_mid_ZN_4': temp_mid_4,
            'Core_mid': temp_core_mid,
        },
        "light_rate": {
            'Perimeter_mid_ZN_1': light_mid_1, 'Perimeter_mid_ZN_2': light_mid_2,
            'Perimeter_mid_ZN_3': light_mid_3, 'Perimeter_mid_ZN_4': light_mid_4,
            'Core_mid': light_core_mid,
        },
        "solar": {
            'Perimeter_mid_ZN_1': solar_mid_1, 'Perimeter_mid_ZN_2': solar_mid_2,
            'Perimeter_mid_ZN_3': solar_mid_3, 'Perimeter_mid_ZN_4': solar_mid_4,
        },
        "ext_irr": {
            'Perimeter_mid_ZN_1': ext_irr_mid_1, 'Perimeter_mid_ZN_2': ext_irr_mid_2,
            'Perimeter_mid_ZN_3': ext_irr_mid_3, 'Perimeter_mid_ZN_4': ext_irr_mid_4,
            'Core_mid': 0.0,
        },
        "energy": {
            'lighting_rate': (light_mid_1 + light_mid_2 + light_mid_3
                              + light_mid_4 + light_core_mid),
            'hvac_demand_rate': hvac_electricity_demand_rate,
        },
        "wpi": current_wpi,
        "temporal": tf,
    }

    sel_g, sel_l, sel_h, sel_c = choose_expert_actions(current_obs, dt)

    for zone, window in WINDOW_ZONES_MID.items():
        eps.actuate_cfs_state(window=window, cfs_state=sel_g[zone])
        eps.actuate_lighting_power(light=zone, value=sel_l[zone])
        eps.actuate_heating_setpoint(zone=zone, value=sel_h[zone])
        eps.actuate_cooling_setpoint(zone=zone, value=sel_c[zone])

        glazing_state_data[zone].append(sel_g[zone])
        lighting_power_data[zone].append(sel_l[zone])
        heating_setpoint_data[zone].append(sel_h[zone])
        cooling_setpoint_data[zone].append(sel_c[zone])

    eps.actuate_lighting_power(light="Core_mid", value=sel_l['Core_mid'])
    eps.actuate_heating_setpoint(zone="Core_mid", value=sel_h['Core_mid'])
    eps.actuate_cooling_setpoint(zone="Core_mid", value=sel_c['Core_mid'])

    lighting_power_data['Core_mid'].append(sel_l['Core_mid'])
    heating_setpoint_data['Core_mid'].append(sel_h['Core_mid'])
    cooling_setpoint_data['Core_mid'].append(sel_c['Core_mid'])

    # ---- SDAR action-mask logging (uses EXECUTED action) ----
    a_vec = encode_action_vec(sel_g, sel_l, sel_h, sel_c)
    prev_a_vec = policy_memory["prev_action_vec"]

    if prev_a_vec is None:
        b_mask = np.ones(ACTION_DIM, dtype=np.float32)
        prev_for_log = a_vec.copy()
    else:
        b_mask = (np.abs(a_vec - prev_a_vec) > MASK_EPS).astype(np.float32)
        prev_for_log = prev_a_vec.copy()

    a_mix = ((1.0 - b_mask) * prev_for_log
             + b_mask * ACTION_MASK_VALUE).astype(np.float32)

    action_vector_data["action"].append(a_vec.copy())
    action_vector_data["prev_action"].append(prev_for_log)
    action_vector_data["selection_mask"].append(b_mask)
    action_vector_data["action_mix"].append(a_mix)

    policy_memory["prev_action_vec"] = a_vec