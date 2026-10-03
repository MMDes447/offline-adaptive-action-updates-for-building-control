"""
EVALUATION CALLBACK â€” trained SDAR-IQL agent in EnergyPlus  (PATCHED)
=====================================================================

Changes vs the previous version:
  * setup() requests the unoccupied-lights-off safeguard explicitly.
  * the learned Bernoulli selector is sampled with a fixed seed, while the
    glazing, lighting, and HVAC proposal actor remains deterministic.
  * agent_data logs the policy action, the executed (safeguarded) action, both
    masks, selector logits/probabilities, and per-step safeguard flags exactly
    once per timestep, so the rerun can be audited.
  * a small end-of-run assertion helper (`verify_safeguard`) checks the
    invariant directly from the logs.

Observation assembly, actuation, and sensor logging are otherwise unchanged.
This rollout intentionally differs in both safeguard execution and selector
realization, and the selector seed makes the result reproducible.
"""

import numpy as np

from sdar_iql_train_3_updated import load_and_act, BEST_CKPT_PATH


# ============================================================
# EVAL CONFIG
# ============================================================

EVAL_CHECKPOINT = "checkpoints/sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3/sdar_iql_sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_augmented_value_standard_iql_beta3_checkpoint_epoch100.pt"

# Sample the learned Bernoulli update selector, but keep the proposed
# glazing/lighting/HVAC actions deterministic.
SELECTOR_MODE = "sample"
PROPOSAL_DETERMINISTIC = True
SELECTOR_SEED = 20260728

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

_AGENT = {"get_action": None, "reset": None, "loaded_from": None}


def setup(checkpoint_path=None):
    """Load the trained agent and reset its internal SDAR/GRU state."""
    path = checkpoint_path if checkpoint_path is not None else EVAL_CHECKPOINT
    get_action, reset = load_and_act(
        path,
        enforce_unoccupied_lights_off=ENFORCE_UNOCCUPIED_LIGHTS_OFF,
        sync_prev_with_deadband=SYNC_PREV_WITH_DEADBAND,
        selector_mode=SELECTOR_MODE,
        proposal_deterministic=PROPOSAL_DETERMINISTIC,
        selector_seed=SELECTOR_SEED,
    )
    reset()
    _AGENT["get_action"] = get_action
    _AGENT["reset"] = reset
    _AGENT["loaded_from"] = str(path)

    eval_state["step"] = 0
    eval_state["glazing"] = {z: INITIAL_GLAZING for z in WINDOW_ZONES_MID}

    for k in agent_data:
        if isinstance(agent_data[k], list):
            agent_data[k].clear()

    print(
        f"[eval callback] agent loaded from {path}  "
        f"(selector_mode={SELECTOR_MODE}, "
        f"proposal_deterministic={PROPOSAL_DETERMINISTIC}, "
        f"selector_seed={SELECTOR_SEED})"
    )
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
    "selection_mask": [],              # policy mask b
    "executed_selection_mask": [],     # mask consistent with executed action
    "policy_executed_action_norm": [], # before safeguard
    "executed_action_norm": [],        # after safeguard (actuated)
    "override_dims": [],               # per-dim: safeguard/deadband changed it
    "lighting_safety_active": [],
    "lighting_safety_override": [],
    "deadband_applied": [],
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
    agent_data["selection_mask"].append(
        np.asarray(result["selection_mask"], dtype=np.float32))
    agent_data["executed_selection_mask"].append(
        np.asarray(result["executed_selection_mask"], dtype=np.float32))
    agent_data["policy_executed_action_norm"].append(
        np.asarray(result["policy_executed_action_norm"], dtype=np.float32))
    agent_data["executed_action_norm"].append(
        np.asarray(result["executed_action_norm"], dtype=np.float32))
    agent_data["override_dims"].append(
        np.asarray(result["override_dims"], dtype=np.float32))
    agent_data["deadband_applied"].append(
        np.asarray(result["deadband_applied"], dtype=np.float32))
    agent_data["lighting_safety_active"].append(bool(result["lighting_safety_active"]))
    agent_data["lighting_safety_override"].append(bool(result["lighting_safety_override"]))


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
