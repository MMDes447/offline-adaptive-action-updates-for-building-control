"""
EnergyPlus evaluation callback for the SDAR execution-policy ablation.

The proposal actor is deterministic in every experiment. Only the execution
strategy changes:

  sample    learned, state-dependent Bernoulli SDAR selector
  constant  state-independent Bernoulli selector with matched mean rates
  periodic  deterministic fractional-period schedule with matched mean rates
  always    every action dimension is reconsidered every timestep

The callback logs the proposal, previous action, policy mask, active and
learned probabilities, safeguarded command, actual physical-change mask, and
all override diagnostics. The same operational safeguards remain active in
every mode.
"""

import os
import numpy as np

from sdar_iql_train_4_updated import load_and_act, BEST_CKPT_PATH


# ============================================================
# EVAL CONFIG
# ============================================================

EVAL_CHECKPOINT = "checkpoints/sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36/sdar_iql_sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_checkpoint_epoch100.pt"

# Set through the environment for batch sweeps, for example:
#   SDAR_SELECTOR_MODE=periodic SDAR_SELECTOR_SEED=20260728
# Valid modes: sample, constant, periodic, always, threshold (legacy).
# SELECTOR_MODE = os.environ.get("SDAR_SELECTOR_MODE", "sample").strip().lower()
SELECTOR_MODE = "periodic"
PROPOSAL_DETERMINISTIC = True
SELECTOR_SEED = int(os.environ.get("SDAR_SELECTOR_SEED", "20260728"))
PERIODIC_PHASE_SEED = int(
    os.environ.get("SDAR_PERIODIC_PHASE_SEED", str(SELECTOR_SEED))
)

# Initial E100 rate-matching values. Replace these with means calculated on a
# validation rollout before producing the final paper results.
CONSTANT_UPDATE_RATES = {
    "glazing": 0.02777,
    "lighting": 0.12440,
    "heating": 0.26803,
    "cooling": 0.26908,
}

# Engineering-unit thresholds used to distinguish a selected reconsideration
# from a physical actuator-command change.
LIGHTING_CHANGE_TOLERANCE_W = 1.0
SETPOINT_CHANGE_TOLERANCE_C = 0.05

# Hard operational constraint: no lighting in an unoccupied building.
ENFORCE_UNOCCUPIED_LIGHTS_OFF = True

# Keep the agent's internal prev_action in sync with the deadband-corrected
# cooling setpoint that is actually actuated. Set False to isolate the
# lighting fix when comparing against the previous rollout.
SYNC_PREV_WITH_DEADBAND = True

OBS_DIM = 29


# ============================================================
# BUILDING CONFIG (mirrors the v12 controller)
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

ZONE_ORDER = list(ZONES_MID)
WINDOW_ORDER = list(WINDOW_ZONES_MID.keys())

LIGHTING_INSTALLED_W = {
    'Perimeter_mid_ZN_1': 2231.0,
    'Perimeter_mid_ZN_2': 2231.0,
    'Perimeter_mid_ZN_3': 1412.0,
    'Perimeter_mid_ZN_4': 1412.0,
    'Core_mid':          10586.0,
}

OCCUPIED_START = 8
OCCUPIED_END = 16
PRECONDITION_LEAD_HRS = 1

INITIAL_GLAZING = 'sr2_ec01'


# ============================================================
# AGENT
# ============================================================

_AGENT = {
    "get_action": None,
    "reset": None,
    "loaded_from": None,
    "selector_mode": None,
    "selector_seed": None,
    "periodic_phase_seed": None,
}


def setup(
    checkpoint_path=None,
    selector_mode=None,
    selector_seed=None,
    periodic_phase_seed=None,
    constant_update_rates=None,
):
    """Load the trained agent and reset all callback and policy state."""
    path = checkpoint_path if checkpoint_path is not None else EVAL_CHECKPOINT
    active_mode = SELECTOR_MODE if selector_mode is None else str(selector_mode).lower()
    active_seed = SELECTOR_SEED if selector_seed is None else int(selector_seed)
    active_phase_seed = (
        PERIODIC_PHASE_SEED
        if periodic_phase_seed is None
        else int(periodic_phase_seed)
    )
    active_rates = (
        dict(CONSTANT_UPDATE_RATES)
        if constant_update_rates is None
        else dict(constant_update_rates)
    )

    get_action, reset = load_and_act(
        path,
        enforce_unoccupied_lights_off=ENFORCE_UNOCCUPIED_LIGHTS_OFF,
        sync_prev_with_deadband=SYNC_PREV_WITH_DEADBAND,
        selector_mode=active_mode,
        proposal_deterministic=PROPOSAL_DETERMINISTIC,
        selector_seed=active_seed,
        constant_update_rates=active_rates,
        periodic_phase_seed=active_phase_seed,
        lighting_change_tolerance_w=LIGHTING_CHANGE_TOLERANCE_W,
        setpoint_change_tolerance_c=SETPOINT_CHANGE_TOLERANCE_C,
    )
    reset()
    _AGENT["get_action"] = get_action
    _AGENT["reset"] = reset
    _AGENT["loaded_from"] = str(path)
    _AGENT["selector_mode"] = active_mode
    _AGENT["selector_seed"] = active_seed
    _AGENT["periodic_phase_seed"] = active_phase_seed

    eval_state["step"] = 0
    eval_state["glazing"] = {z: INITIAL_GLAZING for z in WINDOW_ZONES_MID}
    _clear_all_logs()

    print(
        f"[eval callback] agent loaded from {path}  "
        f"(selector_mode={active_mode}, "
        f"proposal_deterministic={PROPOSAL_DETERMINISTIC}, "
        f"selector_seed={active_seed}, "
        f"periodic_phase_seed={active_phase_seed})"
    )
    if active_mode in {"constant", "periodic"}:
        print(f"[eval callback] fixed update rates = {active_rates}")
    print(f"[eval callback] unoccupied lights-off safeguard = {ENFORCE_UNOCCUPIED_LIGHTS_OFF}")
    return get_action, reset


# ============================================================
# EVAL STATE
# ============================================================

eval_state = {
    "step": 0,
    "glazing": {z: INITIAL_GLAZING for z in WINDOW_ZONES_MID},
}

meter_handles = {
    "Heating:Electricity": None,
    "Cooling:Electricity": None,
    "InteriorLights:Electricity": None,
    "Electricity:HVAC": None,
}


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
}

output_variable_data = {
    'HVAC_electricity_demand_rate': [],
}

agent_data = {
    "observation": [],
    "selection_logits": [],
    "selection_probability": [],
    "learned_selection_logits": [],
    "learned_selection_probability": [],
    "selection_mask": [],              # policy mask b
    "executed_selection_mask": [],     # mask consistent with executed action
    "actual_change_mask": [],          # physical command changed beyond tolerance
    "target_update_rate": [],
    "proposal_action_norm": [],
    "prev_action_norm_before": [],
    "policy_executed_action_norm": [], # before safeguard
    "executed_action_norm": [],        # after safeguard (actuated)
    "override_dims": [],               # per-dim: safeguard/deadband changed it
    "lighting_safety_active": [],
    "lighting_safety_override": [],
    "deadband_applied": [],
    "selector_mode": [],
    "selector_stochastic": [],
    "force_first_update": [],
}


# ============================================================
# HELPERS
# ============================================================

def _clear_list_values(mapping):
    for value in mapping.values():
        if isinstance(value, list):
            value.clear()


def _clear_all_logs():
    """Clear every callback log so repeated runs cannot contaminate results."""
    for mapping in (
        lighting_data,
        temperature_data,
        solar_data,
        ext_irr_data,
        wpi_data,
        glazing_state_data,
        lighting_power_data,
        heating_setpoint_data,
        cooling_setpoint_data,
        meter_data,
        temporal_data,
        output_variable_data,
        agent_data,
    ):
        _clear_list_values(mapping)
    for meter_name in meter_handles:
        meter_handles[meter_name] = None

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


def build_observation(current_obs, tf, hvac_demand_rate):
    """Assemble the 29-dim RAW observation in the training layout/order."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    for i, zone in enumerate(ZONE_ORDER):
        obs[0 + i] = current_obs["temp"][zone]
    for i, zone in enumerate(ZONE_ORDER):
        obs[5 + i] = current_obs["light_rate"][zone]
    for i, zone in enumerate(WINDOW_ORDER):
        obs[10 + i] = current_obs["solar"][zone]
    for i, zone in enumerate(WINDOW_ORDER):
        obs[14 + i] = current_obs["ext_irr"][zone]
    for i, zone in enumerate(WINDOW_ORDER):
        obs[18 + i] = current_obs["wpi"][zone]

    obs[22] = hvac_demand_rate

    obs[23] = tf["hour_sin"]
    obs[24] = tf["hour_cos"]
    obs[25] = tf["doy_sin"]
    obs[26] = tf["doy_cos"]
    obs[27] = float(tf["is_occupied"])
    obs[28] = float(tf["is_preconditioning"])

    return obs


# ============================================================
# CALLBACK
# ============================================================

def callback_func(state):
    if not eps.api.exchange.api_data_fully_ready(state):
        return
    if eps.api.exchange.warmup_flag(state):
        return

    if _AGENT["get_action"] is None:
        setup()

    dt = eps.get_datetime()

    # ---- read sensors ----
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

    # ---- WPI under the glazing currently in effect ----
    current_wpi = {}
    for zone, window in WINDOW_ZONES_MID.items():
        g_now = eval_state["glazing"][zone]
        current_wpi[zone] = float(
            eps.calculate_wpi(zone=zone, cfs_name={window: g_now}).mean()
        )

    # ---- meters ----
    heating_elec = get_meter_value_cached(state, "Heating:Electricity")
    cooling_elec = get_meter_value_cached(state, "Cooling:Electricity")
    lights_elec = get_meter_value_cached(state, "InteriorLights:Electricity")
    hvac_elec = get_meter_value_cached(state, "Electricity:HVAC")

    meter_data['Heating:Electricity'].append(heating_elec)
    meter_data['Cooling:Electricity'].append(cooling_elec)
    meter_data['InteriorLights:Electricity'].append(lights_elec)
    meter_data['Electricity:HVAC'].append(hvac_elec)
    output_variable_data['HVAC_electricity_demand_rate'].append(hvac_electricity_demand_rate)

    # ---- temporal + sensor logging ----
    tf = compute_temporal_features(dt)
    for k, v in tf.items():
        temporal_data[k].append(v)

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
        },
        "wpi": current_wpi,
    }

    raw_obs = build_observation(current_obs, tf, hvac_electricity_demand_rate)

    # ---- query the trained agent ----
    first_step = (eval_state["step"] == 0)
    result = _AGENT["get_action"](
        raw_obs,
        force_first_update=first_step,
    )
    eval_state["step"] += 1

    glz_by_zone = {WINDOW_ORDER[i]: result["glazing"][i] for i in range(len(WINDOW_ORDER))}
    light_by_zone = {ZONE_ORDER[i]: float(result["lighting_power"][i]) for i in range(len(ZONE_ORDER))}
    heat_by_zone = {ZONE_ORDER[i]: float(result["heating_sp"][i]) for i in range(len(ZONE_ORDER))}
    cool_by_zone = {ZONE_ORDER[i]: float(result["cooling_sp"][i]) for i in range(len(ZONE_ORDER))}

    # ---- actuate perimeter zones ----
    for zone, window in WINDOW_ZONES_MID.items():
        eps.actuate_cfs_state(window=window, cfs_state=glz_by_zone[zone])
        eps.actuate_lighting_power(light=zone, value=light_by_zone[zone])
        eps.actuate_heating_setpoint(zone=zone, value=heat_by_zone[zone])
        eps.actuate_cooling_setpoint(zone=zone, value=cool_by_zone[zone])

        glazing_state_data[zone].append(glz_by_zone[zone])
        lighting_power_data[zone].append(light_by_zone[zone])
        heating_setpoint_data[zone].append(heat_by_zone[zone])
        cooling_setpoint_data[zone].append(cool_by_zone[zone])

    # ---- actuate core ----
    core_zone = "Core_mid"
    eps.actuate_lighting_power(light=core_zone, value=light_by_zone[core_zone])
    eps.actuate_heating_setpoint(zone=core_zone, value=heat_by_zone[core_zone])
    eps.actuate_cooling_setpoint(zone=core_zone, value=cool_by_zone[core_zone])

    lighting_power_data[core_zone].append(light_by_zone[core_zone])
    heating_setpoint_data[core_zone].append(heat_by_zone[core_zone])
    cooling_setpoint_data[core_zone].append(cool_by_zone[core_zone])

    for zone in WINDOW_ZONES_MID:
        eval_state["glazing"][zone] = glz_by_zone[zone]

    # ---- agent log ----
    agent_data["observation"].append(raw_obs.copy())
    agent_data["selection_logits"].append(
        np.asarray(
            result["selection_logits"],
            dtype=np.float32,
        ).copy()
    )
    agent_data["selection_probability"].append(
        np.asarray(
            result["selection_probability"],
            dtype=np.float32,
        ).copy()
    )
    agent_data["learned_selection_logits"].append(
        np.asarray(result["learned_selection_logits"], dtype=np.float32).copy()
    )
    agent_data["learned_selection_probability"].append(
        np.asarray(result["learned_selection_probability"], dtype=np.float32).copy()
    )
    agent_data["selection_mask"].append(
        np.asarray(result["selection_mask"], dtype=np.float32).copy())
    agent_data["executed_selection_mask"].append(
        np.asarray(result["executed_selection_mask"], dtype=np.float32).copy())
    agent_data["actual_change_mask"].append(
        np.asarray(result["actual_change_mask"], dtype=np.float32).copy())
    agent_data["target_update_rate"].append(
        np.asarray(result["target_update_rate"], dtype=np.float32).copy())
    agent_data["proposal_action_norm"].append(
        np.asarray(result["proposal_action_norm"], dtype=np.float32).copy())
    agent_data["prev_action_norm_before"].append(
        np.asarray(result["prev_action_norm_before"], dtype=np.float32).copy())
    agent_data["policy_executed_action_norm"].append(
        np.asarray(result["policy_executed_action_norm"], dtype=np.float32).copy())
    agent_data["executed_action_norm"].append(
        np.asarray(result["executed_action_norm"], dtype=np.float32).copy())
    agent_data["override_dims"].append(
        np.asarray(result["override_dims"], dtype=np.float32).copy())
    agent_data["deadband_applied"].append(
        np.asarray(result["deadband_applied"], dtype=np.float32).copy())
    agent_data["lighting_safety_active"].append(bool(result["lighting_safety_active"]))
    agent_data["lighting_safety_override"].append(bool(result["lighting_safety_override"]))
    agent_data["selector_mode"].append(str(result["selector_mode"]))
    agent_data["selector_stochastic"].append(bool(result["selector_stochastic"]))
    agent_data["force_first_update"].append(bool(result["force_first_update"]))


# ============================================================
# POST-RUN VERIFICATION
# ============================================================

def verify_safeguard(verbose=True):
    """Check the invariant directly from the logs. Call after the run."""
    obs = np.array(agent_data["observation"])
    exe = np.array(agent_data["executed_action_norm"])
    pol = np.array(agent_data["policy_executed_action_norm"])
    unocc = obs[:, 27] < 0.5

    light_exe = exe[unocc, 4:9]
    light_pol = pol[unocc, 4:9]
    ok = np.allclose(light_exe, -1.0, atol=1e-6)

    # commanded watts from the actuation log
    lp = np.array([lighting_power_data[z] for z in ZONE_ORDER]).T
    lp_unocc = lp[unocc]
    kwh_unocc = lp_unocc.sum() * (1 / 6) / 1000.0
    kwh_total = lp.sum() * (1 / 6) / 1000.0

    # what the raw policy WOULD have done (counterfactual saving)
    would_w = 0.5 * (light_pol + 1.0)
    installed = np.array([LIGHTING_INSTALLED_W[z] for z in ZONE_ORDER])
    kwh_prevented = (would_w * installed).sum() * (1 / 6) / 1000.0

    if verbose:
        print("=== safeguard verification ===")
        print(f"  unoccupied steps                : {unocc.sum()}")
        print(f"  executed lighting == -1 (0 W)   : {ok}")
        print(f"  override fired on               : {int(np.sum(agent_data['lighting_safety_override']))} steps")
        print(f"  unoccupied lighting energy      : {kwh_unocc:,.1f} kWh   (target 0.0)")
        print(f"  total lighting energy           : {kwh_total:,.1f} kWh")
        print(f"  energy the policy WOULD have used unoccupied: {kwh_prevented:,.1f} kWh")
    return ok, kwh_unocc, kwh_prevented


ACTION_GROUPS = {
    "glazing": slice(0, 4),
    "lighting": slice(4, 9),
    "heating": slice(9, 14),
    "cooling": slice(14, 19),
}


def verify_sdar_logs(verbose=True):
    """Validate shapes, probabilities, and repeat invariants after a run."""
    if not agent_data["observation"]:
        raise RuntimeError("No evaluation data are available.")

    previous = np.asarray(agent_data["prev_action_norm_before"], dtype=np.float32)
    executed = np.asarray(agent_data["executed_action_norm"], dtype=np.float32)
    executed_mask = np.asarray(agent_data["executed_selection_mask"], dtype=np.float32)
    policy_mask = np.asarray(agent_data["selection_mask"], dtype=np.float32)
    active_probability = np.asarray(
        agent_data["selection_probability"], dtype=np.float32
    )
    learned_probability = np.asarray(
        agent_data["learned_selection_probability"], dtype=np.float32
    )

    expected_shape = (len(agent_data["observation"]), 19)
    arrays = {
        "previous": previous,
        "executed": executed,
        "executed_mask": executed_mask,
        "policy_mask": policy_mask,
        "active_probability": active_probability,
        "learned_probability": learned_probability,
    }
    shape_ok = all(value.shape == expected_shape for value in arrays.values())
    probability_ok = bool(
        np.all((active_probability >= 0.0) & (active_probability <= 1.0))
        and np.all((learned_probability >= 0.0) & (learned_probability <= 1.0))
    )
    binary_mask_ok = bool(
        np.all((policy_mask == 0.0) | (policy_mask == 1.0))
        and np.all((executed_mask == 0.0) | (executed_mask == 1.0))
    )

    # If the executed mask says repeat, the final normalized command must be
    # exactly the previous command. Safeguard/deadband overrides are already
    # represented in executed_mask.
    repeat_residual = np.abs(executed - previous) * (1.0 - executed_mask)
    repeat_invariant_ok = bool(np.max(repeat_residual) <= 1e-6)

    result = {
        "shape_ok": shape_ok,
        "probability_ok": probability_ok,
        "binary_mask_ok": binary_mask_ok,
        "repeat_invariant_ok": repeat_invariant_ok,
        "max_repeat_residual": float(np.max(repeat_residual)),
    }
    if verbose:
        print("=== SDAR log verification ===")
        for key, value in result.items():
            print(f"  {key:28s}: {value}")
    return result


def summarize_sdar_ablation(verbose=True):
    """Summarize selected updates, overrides, and real command changes."""
    if not agent_data["observation"]:
        raise RuntimeError("No evaluation data are available.")

    observations = np.asarray(agent_data["observation"], dtype=np.float32)
    selected = np.asarray(agent_data["selection_mask"], dtype=np.float32)
    executed_selected = np.asarray(
        agent_data["executed_selection_mask"], dtype=np.float32
    )
    actual_change = np.asarray(agent_data["actual_change_mask"], dtype=np.float32)
    target_rate = np.asarray(agent_data["target_update_rate"], dtype=np.float32)
    previous = np.asarray(agent_data["prev_action_norm_before"], dtype=np.float32)
    executed = np.asarray(agent_data["executed_action_norm"], dtype=np.float32)
    forced = np.asarray(agent_data["force_first_update"], dtype=bool)

    valid = ~forced
    if not np.any(valid):
        raise RuntimeError("No non-forced evaluation timesteps are available.")

    summary = {
        "selector_mode": _AGENT["selector_mode"],
        "selector_seed": _AGENT["selector_seed"],
        "periodic_phase_seed": _AGENT["periodic_phase_seed"],
        "steps_total": int(len(valid)),
        "steps_analyzed": int(valid.sum()),
        "groups": {},
    }

    for group, action_slice in ACTION_GROUPS.items():
        selected_rate = float(selected[valid, action_slice].mean())
        executed_rate = float(executed_selected[valid, action_slice].mean())
        actual_rate = float(actual_change[valid, action_slice].mean())
        target = float(target_rate[valid, action_slice].mean())

        summary["groups"][group] = {
            "target_rate_pct": 100.0 * target,
            "selected_update_pct": 100.0 * selected_rate,
            "executed_update_pct": 100.0 * executed_rate,
            "actual_change_pct": 100.0 * actual_rate,
            "implied_selected_interval_h": (
                1.0 / (6.0 * selected_rate) if selected_rate > 0.0 else float("inf")
            ),
            "implied_actual_change_interval_h": (
                1.0 / (6.0 * actual_rate) if actual_rate > 0.0 else float("inf")
            ),
        }

    # Selector decisions are conditioned on the current WPI observation. This
    # diagnostic therefore uses obs[:, 18:22], not next-step WPI.
    occupied = observations[:, 27] >= 0.5
    high_wpi = (observations[:, 18:22] > 1000.0) & occupied[:, None] & valid[:, None]
    high_wpi_updates = high_wpi & (selected[:, :4] > 0.5)
    high_wpi_count = int(high_wpi.sum())
    high_wpi_update_rate = (
        100.0 * float(high_wpi_updates.sum()) / high_wpi_count
        if high_wpi_count > 0
        else float("nan")
    )

    previous_glazing_tier = np.rint((previous[:, :4] + 1.0) * 1.5).astype(int)
    executed_glazing_tier = np.rint((executed[:, :4] + 1.0) * 1.5).astype(int)
    darker = high_wpi_updates & (executed_glazing_tier > previous_glazing_tier)
    high_update_count = int(high_wpi_updates.sum())
    darker_given_update = (
        100.0 * float(darker.sum()) / high_update_count
        if high_update_count > 0
        else float("nan")
    )

    summary["high_wpi_glazing"] = {
        "samples": high_wpi_count,
        "selected_update_pct": high_wpi_update_rate,
        "darker_given_update_pct": darker_given_update,
    }

    if verbose:
        print("=== SDAR ablation summary ===")
        print(
            f"  mode={summary['selector_mode']}  "
            f"selector_seed={summary['selector_seed']}  "
            f"periodic_phase_seed={summary['periodic_phase_seed']}"
        )
        print(
            "  group       target%  selected%  executed%  actual-change%  "
            "selected-h  actual-h"
        )
        for group, metrics in summary["groups"].items():
            print(
                f"  {group:10s} "
                f"{metrics['target_rate_pct']:8.3f} "
                f"{metrics['selected_update_pct']:10.3f} "
                f"{metrics['executed_update_pct']:10.3f} "
                f"{metrics['actual_change_pct']:14.3f} "
                f"{metrics['implied_selected_interval_h']:10.3f} "
                f"{metrics['implied_actual_change_interval_h']:8.3f}"
            )
        print(
            "  high-WPI glazing: "
            f"samples={high_wpi_count}, "
            f"selected={high_wpi_update_rate:.3f}%, "
            f"darker|update={darker_given_update:.3f}%"
        )

    return summary
