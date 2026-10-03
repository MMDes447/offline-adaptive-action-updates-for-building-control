"""
Offline RL dataset generator — v8
Meter-free reward + meter-free obs + fixed comfort bands + per-zone lighting
+ per-component reward normalization (unit mean-|abs| then intentional weights)
============================================================================

Drop-in dataset builder to run AFTER an expert EnergyPlus/Sinergym callback
rollout (v12 controller) has filled the global logging dictionaries.

Expected shared globals from the expert callback:
    ZONES_MID, WINDOW_ZONES_MID, AVAILABLE_GLAZING_STATES
    temperature_data, lighting_data, solar_data, ext_irr_data, wpi_data
    meter_data, output_variable_data, temporal_data
    glazing_state_data, lighting_power_data,
    heating_setpoint_data, cooling_setpoint_data
    action_vector_data
"""

import csv
import json
import numpy as np


# ============================================================
# CONFIG
# ============================================================

REWARD_TRANSFORM = "symlog"     # "symlog" | "clip" | "none"
REWARD_CLIP_LIMIT = 50.0
SAVE_REWARD_COMPONENTS = True

REWARD_NORMALIZE = True
REWARD_COMPONENT_CLIP = 10.0

REWARD_IMPORTANCE = {
    "energy_lighting":       0.40,
    "hvac_demand_rate":      1.00,
    "thermal_comfort":       2.00,
    "visual_under":          1.50,
    "visual_over_ctrl":      0.40,
    "visual_over_daylight":  0.20,
    "core_under":            1.00,
    "core_over":             0.40,
    "artificial_excess":     0.25,
    "solar_cooling_penalty": 0.20,
    "solar_heating_bonus":   0.10,
    "switching_penalty":     0.10,
}

TIMEOUT_STEPS = None

# ---- lighting energy from action-derived dimming (nominal power * duty) ----
# Reporting timestep. 6 steps/hour = 10-min steps. Set to match the run.
STEPS_PER_HOUR = 6
TIMESTEP_HOURS = 1.0 / STEPS_PER_HOUR


# ============================================================
# WINDOW MAP
# ============================================================

WINDOW_ZONES_MID = {
    "Perimeter_mid_ZN_1": "Perimeter_mid_ZN_1_Wall_South_Window",
    "Perimeter_mid_ZN_2": "Perimeter_mid_ZN_2_Wall_East_Window",
    "Perimeter_mid_ZN_3": "Perimeter_mid_ZN_3_Wall_North_Window",
    "Perimeter_mid_ZN_4": "Perimeter_mid_ZN_4_Wall_West_Window",
}

GLAZING_TO_ID = {name: i for i, name in enumerate(AVAILABLE_GLAZING_STATES)}


# ============================================================
# ACTION VECTOR LAYOUT
# ============================================================

ACTION_DIM = 19
GLAZE_SLICE = slice(0, 4)
LIGHT_SLICE = slice(4, 9)
HEAT_SLICE = slice(9, 14)
COOL_SLICE = slice(14, 19)

ACTION_MASK_VALUE = -2.0

# Thermal exploration mode ids (must match the exploratory callback).
THERMAL_MODE_TO_ID = {
    "idle": 0,
    "random_inactive_band": 1,
    "active_exploration": 2,
    "expert_recovery": 3,
    "noisy_recovery": 4,
    "sdar_repeat": 5,
}

LIGHTING_INSTALLED_W = {
    "Perimeter_mid_ZN_1": 2231.0,
    "Perimeter_mid_ZN_2": 2231.0,
    "Perimeter_mid_ZN_3": 1412.0,
    "Perimeter_mid_ZN_4": 1412.0,
    "Core_mid":          10586.0,
}


# ============================================================
# REWARD WEIGHTS  (physical; cancel out when REWARD_NORMALIZE=True)
# ============================================================

LIGHTING_TO_LUX = 0.4

ILLUM_TARGET = 500.0
ILLUM_LOW = 450.0
ILLUM_HIGH = 550.0

CORE_TARGET = 500.0
CORE_LOW = 450.0
CORE_HIGH = 550.0

W_LIGHTING = 1.0
W_HVAC_DEMAND = 0.03

W_THERMAL = 10.0
COMFORT_TEMP_LOW = 21.0
COMFORT_TEMP_HIGH = 24.0
COMFORT_OCCUPIED_ONLY = True

W_VISUAL_UNDER = 0.03
W_VISUAL_OVER_CTRL = 0.015
W_VISUAL_OVER_DAYLIGHT = 0.002
W_CORE_UNDER = 0.03
W_CORE_OVER = 0.015

W_ARTIFICIAL_EXCESS = 0.02

W_SOLAR_COOLING = 1e-4
W_SOLAR_HEATING_BONUS = 5e-5

COOLING_ACTIVE_THRESH = 1.0
HEATING_ACTIVE_THRESH = 1.0
SOLAR_IGNORE_THRESH = 50.0
SOLAR_HEAT_CAP = 400.0

W_SWITCH_GLAZING = 0.05
W_SWITCH_LIGHTING = 0.01
W_SWITCH_HVAC = 0.002


# ============================================================
# COLUMN NAMES
# ============================================================

def obs_column_names():
    cols = []
    for zone in ZONES_MID:
        cols.append(f"temp_{zone}")
    for zone in ZONES_MID:
        cols.append(f"light_rate_{zone}")
    for zone in WINDOW_ZONES_MID:
        cols.append(f"solar_{zone}")
    for zone in WINDOW_ZONES_MID:
        cols.append(f"ext_irr_{zone}")
    for zone in WINDOW_ZONES_MID:
        cols.append(f"wpi_{zone}")
    cols += [
        "hvac_electricity_demand_rate",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "is_occupied",
        "is_preconditioning",
    ]
    return cols


def action_column_names():
    cols = []
    for zone in WINDOW_ZONES_MID:
        cols.append(f"glazing_{zone}")
    for zone in ZONES_MID:
        cols.append(f"light_power_{zone}")
    for zone in ZONES_MID:
        cols.append(f"heat_sp_{zone}")
    for zone in ZONES_MID:
        cols.append(f"cool_sp_{zone}")
    return cols


REWARD_COMPONENT_NAMES = [
    "energy_lighting",
    "hvac_demand_rate",
    "thermal_comfort",
    "visual_under",
    "visual_over_ctrl",
    "visual_over_daylight",
    "core_under",
    "core_over",
    "artificial_excess",
    "solar_cooling_penalty",
    "solar_heating_bonus",
    "switching_penalty",
    "raw_total",
    "transformed_total",
]


def _component_names():
    return [n for n in REWARD_COMPONENT_NAMES if n not in ("raw_total", "transformed_total")]


# ============================================================
# BUILDERS
# ============================================================

def build_observation(t):
    obs = []
    for zone in ZONES_MID:
        obs.append(temperature_data[zone][t])
    for zone in ZONES_MID:
        obs.append(lighting_data[zone][t])
    for zone in WINDOW_ZONES_MID:
        obs.append(solar_data[zone][t])
    for zone in WINDOW_ZONES_MID:
        obs.append(ext_irr_data[zone][t])
    for zone in WINDOW_ZONES_MID:
        obs.append(wpi_data[zone][t])
    obs.append(output_variable_data["HVAC_electricity_demand_rate"][t])
    obs.append(temporal_data["hour_sin"][t])
    obs.append(temporal_data["hour_cos"][t])
    obs.append(temporal_data["doy_sin"][t])
    obs.append(temporal_data["doy_cos"][t])
    obs.append(float(temporal_data["is_occupied"][t]))
    obs.append(float(temporal_data["is_preconditioning"][t]))
    if len(obs) != len(obs_column_names()):
        raise ValueError(f"Observation length {len(obs)} != columns {len(obs_column_names())}")
    return obs


def build_action_raw(t):
    act = []
    for zone in WINDOW_ZONES_MID:
        act.append(GLAZING_TO_ID[glazing_state_data[zone][t]])
    for zone in ZONES_MID:
        act.append(lighting_power_data[zone][t])
    for zone in ZONES_MID:
        act.append(heating_setpoint_data[zone][t])
    for zone in ZONES_MID:
        act.append(cooling_setpoint_data[zone][t])
    if len(act) != ACTION_DIM:
        raise ValueError(f"Action length {len(act)} != ACTION_DIM {ACTION_DIM}")
    return act


# ============================================================
# LIGHTING ENERGY FROM ACTION-DERIVED DIMMING
# ============================================================
# The per-zone normalized lighting action a in [-1, 1] maps exactly to a
# dimming duty u = (a + 1) / 2 in [0, 1]. Energy is therefore reconstructed as
#   E = sum_z ( nominal_W[z] * u[z] ) * dt_hours / 1000   (kWh)
# using the action at time t (energy is a property of a_t in (s_t, a_t, r_t)).
# This does NOT use meters or the logged Lights Electricity Rate; those remain
# available as observations and for validation:  P_EnergyPlus ~= sum_z P_nom*u.


def compute_lighting_energy_kwh(t):
    action_t = np.asarray(
        action_vector_data["action"][t],
        dtype=np.float64,
    )

    # Convert normalized [-1, 1] actions to dimming [0, 1]
    dimming = np.clip(
        (action_t[LIGHT_SLICE] + 1.0) / 2.0,
        0.0,
        1.0,
    )

    nominal_w = np.array(
        [LIGHTING_INSTALLED_W[zone] for zone in ZONES_MID],
        dtype=np.float64,
    )

    zone_power_w = nominal_w * dimming
    total_power_w = zone_power_w.sum()

    energy_kwh = (
        total_power_w
        * TIMESTEP_HOURS
        / 1000.0
    )

    return float(energy_kwh)


# ============================================================
# REWARD — components + scalar
# ============================================================

def _symlog(x):
    return float(np.sign(x) * np.log1p(np.abs(x)))


def _transform(r):
    if REWARD_TRANSFORM == "symlog":
        return _symlog(r)
    if REWARD_TRANSFORM == "clip":
        return float(np.clip(r, -REWARD_CLIP_LIMIT, REWARD_CLIP_LIMIT))
    if REWARD_TRANSFORM == "none":
        return float(r)
    raise ValueError(f"Unknown REWARD_TRANSFORM={REWARD_TRANSFORM!r}")


def _transform_vec(r):
    r = np.asarray(r, dtype=np.float64)
    if REWARD_TRANSFORM == "symlog":
        return (np.sign(r) * np.log1p(np.abs(r))).astype(np.float32)
    if REWARD_TRANSFORM == "clip":
        return np.clip(r, -REWARD_CLIP_LIMIT, REWARD_CLIP_LIMIT).astype(np.float32)
    if REWARD_TRANSFORM == "none":
        return r.astype(np.float32)
    raise ValueError(f"Unknown REWARD_TRANSFORM={REWARD_TRANSFORM!r}")


def compute_reward_components(t, selection_mask_t):
    occupied_next = bool(temporal_data["is_occupied"][t + 1])
    # Visual comfort is a property of the lighting ACTION at t, so gate it on
    # occupancy at t (avoids the artificial under-lighting penalty at the 08:00
    # transition where occupancy flips between t and t+1).
    occupied_lighting = bool(temporal_data["is_occupied"][t])

    hvac_demand_w = output_variable_data["HVAC_electricity_demand_rate"][t + 1]
    r_hvac_demand = -W_HVAC_DEMAND * (hvac_demand_w / 1000.0)

    # lighting energy from action-derived dimming at time t (nominal * duty)
    lighting_energy_kwh = compute_lighting_energy_kwh(t)
    e_light = -W_LIGHTING * lighting_energy_kwh

    thermal_penalty = 0.0
    if (not COMFORT_OCCUPIED_ONLY) or occupied_next:
        for zone in ZONES_MID:
            temp_next = temperature_data[zone][t + 1]
            if temp_next < COMFORT_TEMP_LOW:
                thermal_penalty += COMFORT_TEMP_LOW - temp_next
            elif temp_next > COMFORT_TEMP_HIGH:
                thermal_penalty += temp_next - COMFORT_TEMP_HIGH
    r_thermal = -W_THERMAL * thermal_penalty

    visual_under_p = 0.0
    visual_over_ctrl_p = 0.0
    visual_over_daylight_p = 0.0
    artificial_excess_p = 0.0
    core_under_p = 0.0
    core_over_p = 0.0

    if occupied_lighting:
        for zone in WINDOW_ZONES_MID:
            daylight_lux = wpi_data[zone][t + 1]
            light_power = lighting_power_data[zone][t]
            artificial_lux = light_power * LIGHTING_TO_LUX
            total_lux = daylight_lux + artificial_lux

            if total_lux < ILLUM_LOW:
                visual_under_p += ILLUM_LOW - total_lux
            elif total_lux > ILLUM_HIGH:
                if daylight_lux >= ILLUM_HIGH:
                    visual_over_daylight_p += np.log1p(daylight_lux - ILLUM_HIGH)
                    visual_over_ctrl_p += artificial_lux
                else:
                    visual_over_ctrl_p += total_lux - ILLUM_HIGH

            needed_art = max(0.0, ILLUM_TARGET - min(daylight_lux, ILLUM_TARGET))
            artificial_excess_p += max(0.0, artificial_lux - needed_art)

        core_power = lighting_power_data["Core_mid"][t]
        core_lux = core_power * LIGHTING_TO_LUX
        core_under_p = max(0.0, CORE_LOW - core_lux)
        core_over_p = max(0.0, core_lux - CORE_HIGH)
        artificial_excess_p += max(0.0, core_lux - CORE_TARGET)

    r_visual_under = -W_VISUAL_UNDER * visual_under_p
    r_visual_over_ctrl = -W_VISUAL_OVER_CTRL * visual_over_ctrl_p
    r_visual_over_daylight = -W_VISUAL_OVER_DAYLIGHT * visual_over_daylight_p
    r_core_under = -W_CORE_UNDER * core_under_p
    r_core_over = -W_CORE_OVER * core_over_p
    r_artificial_excess = -W_ARTIFICIAL_EXCESS * artificial_excess_p

    heating_flag = meter_data["Heating:Electricity"][t + 1] > HEATING_ACTIVE_THRESH
    cooling_flag = meter_data["Cooling:Electricity"][t + 1] > COOLING_ACTIVE_THRESH

    solar_cooling_p = 0.0
    solar_heating_b = 0.0
    for zone in WINDOW_ZONES_MID:
        solar_next = solar_data[zone][t + 1]
        if cooling_flag:
            if solar_next > SOLAR_IGNORE_THRESH:
                solar_cooling_p += solar_next - SOLAR_IGNORE_THRESH
        elif heating_flag:
            solar_heating_b += min(solar_next, SOLAR_HEAT_CAP)

    r_solar_cool = -W_SOLAR_COOLING * solar_cooling_p
    r_solar_heat = +W_SOLAR_HEATING_BONUS * solar_heating_b

    if t == 0:
        r_switch = 0.0
    else:
        b = selection_mask_t
        n_glaze = float(b[GLAZE_SLICE].sum())
        n_light = float(b[LIGHT_SLICE].sum())
        n_hvac = float(b[HEAT_SLICE].sum() + b[COOL_SLICE].sum())
        r_switch = -(
            W_SWITCH_GLAZING * n_glaze
            + W_SWITCH_LIGHTING * n_light
            + W_SWITCH_HVAC * n_hvac
        )

    raw_total = (
        e_light + r_hvac_demand + r_thermal
        + r_visual_under + r_visual_over_ctrl + r_visual_over_daylight
        + r_core_under + r_core_over + r_artificial_excess
        + r_solar_cool + r_solar_heat + r_switch
    )

    return {
        "energy_lighting": e_light,
        "hvac_demand_rate": r_hvac_demand,
        "thermal_comfort": r_thermal,
        "visual_under": r_visual_under,
        "visual_over_ctrl": r_visual_over_ctrl,
        "visual_over_daylight": r_visual_over_daylight,
        "core_under": r_core_under,
        "core_over": r_core_over,
        "artificial_excess": r_artificial_excess,
        "solar_cooling_penalty": r_solar_cool,
        "solar_heating_bonus": r_solar_heat,
        "switching_penalty": r_switch,
        "raw_total": raw_total,
        "transformed_total": _transform(raw_total),
    }


def normalize_reward_components(components_raw, comp_names, component_scales=None):
    C = len(comp_names)
    importance_vec = np.array([REWARD_IMPORTANCE.get(nm, 1.0) for nm in comp_names],
                              dtype=np.float64)

    if component_scales is not None:
        comp_scales = np.array([float(component_scales[nm]) for nm in comp_names],
                               dtype=np.float64)
        comp_scales[comp_scales < 1e-8] = 1.0
        scales_source = "external (shared across experts)"
        apply = True
    elif REWARD_NORMALIZE:
        comp_scales = np.abs(components_raw).mean(axis=0).astype(np.float64)
        comp_scales[comp_scales < 1e-8] = 1.0
        scales_source = "dataset-local mean|abs|"
        apply = True
    else:
        comp_scales = np.ones(C, dtype=np.float64)
        scales_source = "none (raw physical W_* weights)"
        apply = False

    if apply:
        components_final = (components_raw.astype(np.float64) / comp_scales) * importance_vec
    else:
        components_final = components_raw.astype(np.float64).copy()

    if REWARD_COMPONENT_CLIP is not None:
        components_final = np.clip(components_final,
                                   -REWARD_COMPONENT_CLIP, REWARD_COMPONENT_CLIP)

    return components_final.astype(np.float32), comp_scales, importance_vec, scales_source


# ============================================================
# DIAGNOSTICS
# ============================================================

def _print_reward_diagnostics(rewards_raw, rewards_tx, components_arr):
    print("\n=== Reward diagnostics ===")
    qs = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
    print(f"  raw:         mean={rewards_raw.mean():+.4f}  std={rewards_raw.std():.4f}")
    print(f"               min={rewards_raw.min():+.4f}    max={rewards_raw.max():+.4f}")
    print(f"               quantiles {qs} = "
          f"{[round(float(q), 3) for q in np.quantile(rewards_raw, qs)]}")
    print(f"  transformed: mean={rewards_tx.mean():+.4f}  std={rewards_tx.std():.4f}")
    print(f"               min={rewards_tx.min():+.4f}    max={rewards_tx.max():+.4f}")
    print(f"               quantiles {qs} = "
          f"{[round(float(q), 3) for q in np.quantile(rewards_tx, qs)]}")
    print("\n=== Per-component mean |contribution| (final, post-normalization) ===")
    comp_names = _component_names()
    for i, name in enumerate(comp_names):
        col = components_arr[:, i]
        print(f"  {name:24s}  mean={col.mean():+.6f}  mean|.|={np.abs(col).mean():.6f}  "
              f"min={col.min():+.4f}  max={col.max():+.4f}")
    comp_sum = components_arr.sum(axis=1)
    err = comp_sum - rewards_raw
    print("\n=== Component sum check ===")
    print(f"  mean abs(sum_components - raw_total): {np.mean(np.abs(err)):.8f}")
    print(f"  max  abs(sum_components - raw_total): {np.max(np.abs(err)):.8f}")


def _print_normalization(comp_names, comp_scales, importance_vec, scales_source):
    print("\n=== Reward component normalization ===")
    print(f"  REWARD_NORMALIZE={REWARD_NORMALIZE}  clip={REWARD_COMPONENT_CLIP}  "
          f"scales={scales_source}")
    print(f"  {'component':24s}  {'scale(mean|abs|)':>16s}  {'importance':>10s}  "
          f"{'eff_weight':>10s}")
    for nm, sc, imp in zip(comp_names, comp_scales, importance_vec):
        eff = imp / sc if sc != 0 else 0.0
        print(f"  {nm:24s}  {sc:16.6f}  {imp:10.3f}  {eff:10.4f}")


def _print_thermal_violation_rate():
    n_steps = len(temperature_data[ZONES_MID[0]])
    violations = 0
    total = 0
    for t in range(n_steps):
        occ = bool(temporal_data["is_occupied"][t])
        if COMFORT_OCCUPIED_ONLY and not occ:
            continue
        for zone in ZONES_MID:
            tmp = temperature_data[zone][t]
            total += 1
            if tmp < COMFORT_TEMP_LOW or tmp > COMFORT_TEMP_HIGH:
                violations += 1
    rate = 100.0 * violations / max(total, 1)
    scope = "occupied zone-steps" if COMFORT_OCCUPIED_ONLY else "all zone-steps"
    print("\n=== Thermal violation rate (fixed band "
          f"{COMFORT_TEMP_LOW}-{COMFORT_TEMP_HIGH}°C) ===")
    print(f"  {violations}/{total} {scope} outside band ({rate:.2f}%)")


def _print_switching_diagnostics(selection_masks):
    print("\n=== SDAR selection mask diagnostics ===")
    print("  b=1 means dim acted/changed, b=0 means dim repeated")
    groups = {
        "glazing  (dims 0:4)": selection_masks[:, GLAZE_SLICE],
        "lighting (dims 4:9)": selection_masks[:, LIGHT_SLICE],
        "heating  (dims 9:14)": selection_masks[:, HEAT_SLICE],
        "cooling  (dims 14:19)": selection_masks[:, COOL_SLICE],
        "overall  (all 19 dims)": selection_masks,
    }
    for name, m in groups.items():
        mean_b = float(m.mean())
        apr = 1.0 / max(mean_b, 1e-9)
        acted_dims = float(m.sum(axis=1).mean())
        print(f"  {name:24s}  mean(b)={mean_b:.4f}  APR={apr:7.2f}  acted_dims/step={acted_dims:.3f}")


def _print_scale_check(observations, actions_raw, actions_norm):
    print("\n=== Observation column ranges ===")
    cols = obs_column_names()
    for i, name in enumerate(cols):
        col = observations[:, i]
        print(f"  [{i:2d}] {name:40s} mean={col.mean():+.3f}  std={col.std():.3f}  "
              f"min={col.min():+.2f}  max={col.max():+.2f}")
    print("\n=== Action column ranges (raw / physical units) ===")
    a_cols = action_column_names()
    for i, name in enumerate(a_cols):
        col = actions_raw[:, i]
        print(f"  [{i:2d}] {name:40s} mean={col.mean():+.3f}  std={col.std():.3f}  "
              f"min={col.min():+.2f}  max={col.max():+.2f}")
    print("\n=== Action column ranges (normalized [-1, 1], SDAR space) ===")
    for i, name in enumerate(a_cols):
        col = actions_norm[:, i]
        print(f"  [{i:2d}] {name:40s} mean={col.mean():+.3f}  std={col.std():.3f}  "
              f"min={col.min():+.2f}  max={col.max():+.2f}")


def _print_hvac_demand_diagnostics(observations):
    cols = obs_column_names()
    hvac_idx = cols.index("hvac_electricity_demand_rate")
    occ_idx = cols.index("is_occupied")
    pre_idx = cols.index("is_preconditioning")
    hvac = observations[:, hvac_idx]
    occupied = observations[:, occ_idx] > 0.5
    pre = observations[:, pre_idx] > 0.5
    setback = (~occupied) & (~pre)
    print("\n=== HVAC demand-rate diagnostics ===")
    for name, mask in {
        "all": np.ones(len(hvac), dtype=bool),
        "occupied": occupied,
        "preconditioning": pre,
        "setback": setback,
    }.items():
        if mask.sum() == 0:
            print(f"  {name:16s}: no samples")
            continue
        x = hvac[mask]
        print(f"  {name:16s}: mean={x.mean():10.2f} W  p95={np.quantile(x, 0.95):10.2f} W  "
              f"max={x.max():10.2f} W")


def _make_timeouts(T):
    timeouts = np.zeros(T, dtype=np.float32)
    if TIMEOUT_STEPS is None:
        return timeouts
    if TIMEOUT_STEPS <= 0:
        return timeouts
    for i in range(T):
        if (i + 1) % int(TIMEOUT_STEPS) == 0:
            timeouts[i] = 1.0
    return timeouts


# ============================================================
# SHARED REWARD SCALES (compute ONCE across all experts, reuse)
# ============================================================
# For pooled multi-expert offline RL, do NOT let each dataset self-normalize.
# 1) Build every expert's raw components with build_offline_dataset(..., component_scales=None)
#    and grab result["reward_components_raw"] (also saved in each NPZ).
# 2) Stack them and call compute_shared_scales() to get ONE scale dict.
# 3) Re-run every expert with build_offline_dataset(..., component_scales=SHARED).

def compute_shared_scales(components_raw_list, comp_names=None):
    """
    components_raw_list : list of (T_i, C) arrays of RAW reward components
                          (result["reward_components_raw"] from each expert).
    Returns {component_name: scale} using pooled mean-|abs| across all experts.
    """
    if comp_names is None:
        comp_names = _component_names()
    stacked = np.concatenate([np.asarray(a, dtype=np.float64)
                              for a in components_raw_list], axis=0)
    if stacked.shape[1] != len(comp_names):
        raise ValueError(
            f"components have {stacked.shape[1]} cols but {len(comp_names)} names")
    scales = np.abs(stacked).mean(axis=0)
    scales[scales < 1e-8] = 1.0
    return {nm: float(sc) for nm, sc in zip(comp_names, scales)}


# ============================================================
# MAIN BUILDER
# ============================================================

def _default_prefix():
    """Unique per-rollout filename so episodes don't overwrite each other."""
    try:
        return (
            f"offline_{EXPERT_MODE}_"
            f"{exploration_data.get('noise_profile', NOISE_PROFILE)}_"
            f"episode_{exploration_data.get('episode_id', EPISODE_ID):03d}"
        )
    except Exception:
        return "offline_sdar_rl_replay_dataset"


def build_offline_dataset(
    save_npz=True,
    save_csv=True,
    prefix=None,
    component_scales=None,
):
    if prefix is None:
        prefix = _default_prefix()
    n = len(temperature_data[ZONES_MID[0]])
    T = n - 1

    expected_lengths = {
        "temperature_data": len(temperature_data[ZONES_MID[0]]),
        "lighting_data": len(lighting_data[ZONES_MID[0]]),
        "solar_data": len(solar_data[list(WINDOW_ZONES_MID.keys())[0]]),
        "ext_irr_data": len(ext_irr_data[list(WINDOW_ZONES_MID.keys())[0]]),
        "wpi_data": len(wpi_data[list(WINDOW_ZONES_MID.keys())[0]]),
        "meter_heating": len(meter_data["Heating:Electricity"]),
        "meter_cooling": len(meter_data["Cooling:Electricity"]),
        "hvac_demand_rate": len(output_variable_data["HVAC_electricity_demand_rate"]),
        "glazing_state": len(glazing_state_data[list(WINDOW_ZONES_MID.keys())[0]]),
        "lighting_power": len(lighting_power_data[ZONES_MID[0]]),
        "heating_setpoint": len(heating_setpoint_data[ZONES_MID[0]]),
        "cooling_setpoint": len(cooling_setpoint_data[ZONES_MID[0]]),
        "temporal_hour_sin": len(temporal_data["hour_sin"]),
        "temporal_is_occupied": len(temporal_data["is_occupied"]),
        "sdar_action": len(action_vector_data["action"]),
        "sdar_prev_action": len(action_vector_data["prev_action"]),
        "sdar_mask": len(action_vector_data["selection_mask"]),
        "sdar_mix": len(action_vector_data["action_mix"]),
        "exploration_expert_action": len(exploration_data["expert_action"]),
        "exploration_executed_action": len(exploration_data["executed_action"]),
        "exploration_mask": len(exploration_data["exploration_mask"]),
        "exploration_glazing": len(exploration_data["glazing_explored"]),
        "exploration_lighting": len(exploration_data["lighting_explored"]),
        "exploration_thermal_mode": len(exploration_data["thermal_mode"]),
        "exploration_epsilon": len(exploration_data["epsilon"]),
        "exploration_sdar_forced_repeat": len(exploration_data["sdar_forced_repeat"]),
    }

    print("=== Length sanity check ===")
    bad_lengths = []
    for k, v in expected_lengths.items():
        status = "OK" if v == n else f"MISMATCH (expected {n})"
        print(f"  {k}: {v}  {status}")
        if v != n:
            bad_lengths.append((k, v))
    if bad_lengths:
        raise ValueError(f"Length mismatch in logged arrays: {bad_lengths}")

    observations = np.array([build_observation(t) for t in range(T)], dtype=np.float32)
    next_observations = np.array([build_observation(t + 1) for t in range(T)], dtype=np.float32)

    obs_dim = len(obs_column_names())
    if observations.shape[1] != obs_dim:
        raise ValueError(f"Expected {obs_dim}-dim observations, got {observations.shape}")

    actions_raw = np.array([build_action_raw(t) for t in range(T)], dtype=np.float32)

    sdar_actions = np.asarray(action_vector_data["action"], dtype=np.float32)[:T]
    sdar_prev_actions = np.asarray(action_vector_data["prev_action"], dtype=np.float32)[:T]
    sdar_masks = np.asarray(action_vector_data["selection_mask"], dtype=np.float32)[:T]
    sdar_mixes = np.asarray(action_vector_data["action_mix"], dtype=np.float32)[:T]

    for name, arr in [
        ("sdar_actions", sdar_actions),
        ("sdar_prev_actions", sdar_prev_actions),
        ("sdar_masks", sdar_masks),
        ("sdar_mixes", sdar_mixes),
    ]:
        if arr.ndim != 2 or arr.shape != (T, ACTION_DIM):
            raise ValueError(
                f"{name} has shape {arr.shape}, expected ({T}, {ACTION_DIM}). "
                "Did the callback use the 19-dim SDAR layout?"
            )

    # ---- #5: primary action is the EXECUTED action; verify it matches SDAR ----
    executed_actions = np.asarray(
        exploration_data["executed_action"], dtype=np.float32)[:T]
    if not np.allclose(executed_actions, sdar_actions, atol=1e-6):
        raise ValueError(
            "Executed exploration actions do not match SDAR actions.")
    actions = sdar_actions  # never train on expert_action

    # #2: hard guard — normalized actions must lie within [-1, 1].
    if np.any(actions < -1.0001) or np.any(actions > 1.0001):
        raise ValueError(
            f"Normalized actions outside [-1,1]: "
            f"min={actions.min()}, max={actions.max()}")

    # ---- #7: expert action + exploration mask ----
    expert_actions = np.asarray(
        exploration_data["expert_action"], dtype=np.float32)[:T]
    exploration_masks = np.asarray(
        exploration_data["exploration_mask"], dtype=np.float32)[:T]
    sdar_forced_repeat = np.asarray(
        exploration_data["sdar_forced_repeat"], dtype=np.float32)[:T]

    # per-zone explored flags -> arrays
    glazing_explored = np.array([
        [float(exploration_data["glazing_explored"][t][zone])
         for zone in WINDOW_ZONES_MID]
        for t in range(T)], dtype=np.float32)
    lighting_explored = np.array([
        [float(exploration_data["lighting_explored"][t][zone])
         for zone in ZONES_MID]
        for t in range(T)], dtype=np.float32)

    # thermal modes -> integer ids
    thermal_mode_ids = np.array([
        [THERMAL_MODE_TO_ID[exploration_data["thermal_mode"][t][zone]]
         for zone in ZONES_MID]
        for t in range(T)], dtype=np.int8)

    # ---- #2: physical lighting quantities from executed dimming ----
    lighting_dimming = np.array([
        np.clip((np.asarray(action_vector_data["action"][t])[LIGHT_SLICE] + 1.0) / 2.0,
                0.0, 1.0)
        for t in range(T)], dtype=np.float32)
    nominal_power_w = np.array(
        [LIGHTING_INSTALLED_W[zone] for zone in ZONES_MID], dtype=np.float32)
    lighting_power_from_action_w = (lighting_dimming * nominal_power_w[None, :]).astype(np.float32)
    lighting_energy_kwh = (
        lighting_power_from_action_w.sum(axis=1) * TIMESTEP_HOURS / 1000.0
    ).astype(np.float32)

    comp_names = _component_names()
    components_raw = np.zeros((T, len(comp_names)), dtype=np.float32)
    for t in range(T):
        comps = compute_reward_components(t, sdar_masks[t])
        for i, name in enumerate(comp_names):
            components_raw[t, i] = comps[name]

    components_final, comp_scales, importance_vec, scales_source = \
        normalize_reward_components(components_raw, comp_names, component_scales)

    rewards_raw = components_final.sum(axis=1).astype(np.float32)
    rewards_tx = _transform_vec(rewards_raw)
    rewards = rewards_tx.astype(np.float32)

    terminals = np.zeros(T, dtype=np.float32)
    terminals[-1] = 1.0
    timeouts = _make_timeouts(T)

    obs_mean = observations.mean(axis=0).astype(np.float32)
    obs_std = observations.std(axis=0).astype(np.float32) + 1e-6
    act_mean = actions.mean(axis=0).astype(np.float32)
    act_std = actions.std(axis=0).astype(np.float32) + 1e-6

    light_ranges = np.array(
        [[0.0, LIGHTING_INSTALLED_W[zone]] for zone in ZONES_MID], dtype=np.float32
    )

    print("\n=== Dataset shapes ===")
    print(f"  observations:      {observations.shape}  (29-dim, meter-free)")
    print(f"  actions  (norm):   {actions.shape}      (primary, SDAR-ready, per-zone lighting norm)")
    print(f"  actions_raw:       {actions_raw.shape}  (physical units)")
    print(f"  prev_actions:      {sdar_prev_actions.shape}")
    print(f"  selection_masks:   {sdar_masks.shape}")
    print(f"  action_mixes:      {sdar_mixes.shape}")
    print(f"  rewards:           {rewards.shape}  (transform={REWARD_TRANSFORM}, normalize={REWARD_NORMALIZE})")
    print(f"  rewards_raw:       {rewards_raw.shape}  (balanced sum, pre-transform)")
    print(f"  next_observations: {next_observations.shape}")
    print(f"  terminals:         {terminals.shape}  true terminals={int(terminals.sum())}")
    print(f"  timeouts:          {timeouts.shape}   artificial timeouts={int(timeouts.sum())}")
    if SAVE_REWARD_COMPONENTS:
        print(f"  reward_components: {components_final.shape}  ({len(comp_names)} terms, final)")

    _print_normalization(comp_names, comp_scales, importance_vec, scales_source)
    _print_reward_diagnostics(rewards_raw, rewards_tx, components_final)
    _print_thermal_violation_rate()
    _print_switching_diagnostics(sdar_masks)
    _print_hvac_demand_diagnostics(observations)
    _print_scale_check(observations, actions_raw, actions)

    if save_npz:
        npz_path = f"{prefix}.npz"
        hvac_idx = obs_column_names().index("hvac_electricity_demand_rate")
        hvac_demand_rate = observations[:, hvac_idx].astype(np.float32)
        next_hvac_demand_rate = next_observations[:, hvac_idx].astype(np.float32)

        save_dict = dict(
            observations=observations,
            actions=actions,
            actions_raw=actions_raw,
            prev_actions=sdar_prev_actions,
            selection_masks=sdar_masks,
            action_mixes=sdar_mixes,
            rewards=rewards,
            rewards_raw=rewards_raw,
            next_observations=next_observations,
            terminals=terminals,
            timeouts=timeouts,
            hvac_demand_rate=hvac_demand_rate,
            next_hvac_demand_rate=next_hvac_demand_rate,
            obs_mean=obs_mean,
            obs_std=obs_std,
            act_mean=act_mean,
            act_std=act_std,
            obs_cols=np.array(obs_column_names()),
            act_cols=np.array(action_column_names()),
            reward_transform=np.array(REWARD_TRANSFORM),
            reward_normalize=np.array(REWARD_NORMALIZE),
            reward_component_clip=np.array(
                -1.0 if REWARD_COMPONENT_CLIP is None else REWARD_COMPONENT_CLIP,
                dtype=np.float32),
            action_mask_value=np.float32(ACTION_MASK_VALUE),
            # exploration provenance
            expert_actions=expert_actions,
            executed_actions=executed_actions,
            exploration_masks=exploration_masks,
            glazing_explored=glazing_explored,
            lighting_explored=lighting_explored,
            thermal_mode_ids=thermal_mode_ids,
            exploration_seed=np.int64(exploration_data["seed"]),
            exploration_epsilon=np.asarray(
                exploration_data["epsilon"], dtype=np.float32)[:T],
            sdar_forced_repeat=sdar_forced_repeat,
            episode_id=np.int64(exploration_data.get("episode_id", -1)),
            noise_profile=np.array(str(exploration_data.get("noise_profile", "unknown"))),
            # physical lighting quantities (action-derived)
            lighting_dimming=lighting_dimming,
            lighting_power_from_action_w=lighting_power_from_action_w,
            lighting_energy_kwh=lighting_energy_kwh,
            light_ranges_per_zone=light_ranges,
            light_range_zone_order=np.array(list(ZONES_MID)),
            action_layout=np.array(
                {
                    "glazing": [0, 4],
                    "lighting": [4, 9],
                    "heating": [9, 14],
                    "cooling": [14, 19],
                },
                dtype=object,
            ),
        )

        if SAVE_REWARD_COMPONENTS:
            save_dict["reward_components"] = components_final
            save_dict["reward_components_raw"] = components_raw
            save_dict["reward_component_names"] = np.array(comp_names)
            save_dict["reward_component_scales"] = comp_scales.astype(np.float32)
            save_dict["reward_component_importance"] = importance_vec.astype(np.float32)

        if "expert_id" in temporal_data:
            save_dict["expert_ids"] = np.asarray(temporal_data["expert_id"], dtype=np.int64)[:T]

        np.savez_compressed(npz_path, **save_dict)
        print(f"\nSaved NPZ: {npz_path}")

        stats_path = f"{prefix}_stats.json"
        with open(stats_path, "w") as f:
            json.dump(
                {
                    "reward_transform": REWARD_TRANSFORM,
                    "reward_normalize": bool(REWARD_NORMALIZE),
                    "reward_component_clip": REWARD_COMPONENT_CLIP,
                    "reward_scales_source": scales_source,
                    "lighting_and_hvac_energy_terms_are_meter_free": True,
                    "obs_is_meter_free": True,
                    "n_transitions": int(T),
                    "obs_dim": int(observations.shape[1]),
                    "act_dim": int(actions.shape[1]),
                    "obs_cols": obs_column_names(),
                    "act_cols": action_column_names(),
                    "reward_raw_mean": float(rewards_raw.mean()),
                    "reward_raw_std": float(rewards_raw.std()),
                    "reward_raw_min": float(rewards_raw.min()),
                    "reward_raw_max": float(rewards_raw.max()),
                    "reward_tx_mean": float(rewards_tx.mean()),
                    "reward_tx_std": float(rewards_tx.std()),
                    "reward_tx_min": float(rewards_tx.min()),
                    "reward_tx_max": float(rewards_tx.max()),
                    "reward_component_names": comp_names,
                    "reward_component_scales": comp_scales.tolist(),
                    "reward_component_importance": importance_vec.tolist(),
                    "reward_importance_config": REWARD_IMPORTANCE,
                    "comfort_band": {
                        "low_c": COMFORT_TEMP_LOW,
                        "high_c": COMFORT_TEMP_HIGH,
                        "occupied_only": COMFORT_OCCUPIED_ONLY,
                    },
                    "mean_selection_mask_overall": float(sdar_masks.mean()),
                    "mean_selection_mask_glazing": float(sdar_masks[:, GLAZE_SLICE].mean()),
                    "mean_selection_mask_lighting": float(sdar_masks[:, LIGHT_SLICE].mean()),
                    "mean_selection_mask_heating": float(sdar_masks[:, HEAT_SLICE].mean()),
                    "mean_selection_mask_cooling": float(sdar_masks[:, COOL_SLICE].mean()),
                    "hvac_demand_rate_mean_w": float(hvac_demand_rate.mean()),
                    "hvac_demand_rate_p95_w": float(np.quantile(hvac_demand_rate, 0.95)),
                    "hvac_demand_rate_max_w": float(hvac_demand_rate.max()),
                    "light_installed_w": {z: LIGHTING_INSTALLED_W[z] for z in ZONES_MID},
                    "timeout_steps": None if TIMEOUT_STEPS is None else int(TIMEOUT_STEPS),
                    "num_timeouts": int(timeouts.sum()),
                    "obs_mean": obs_mean.tolist(),
                    "obs_std": obs_std.tolist(),
                    "act_mean": act_mean.tolist(),
                    "act_std": act_std.tolist(),
                },
                f,
                indent=2,
            )
        print(f"Saved stats: {stats_path}")

    if save_csv:
        csv_path = f"{prefix}.csv"
        obs_cols = obs_column_names()
        act_cols = action_column_names()
        mask_cols = [f"mask_{c}" for c in act_cols]
        next_obs_cols = [f"next_{c}" for c in obs_cols]

        header = (
            obs_cols
            + act_cols
            + mask_cols
            + ["reward", "reward_raw"]
            + (comp_names if SAVE_REWARD_COMPONENTS else [])
            + next_obs_cols
            + ["terminal", "timeout"]
        )

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for t in range(T):
                row = (
                    observations[t].tolist()
                    + actions_raw[t].tolist()
                    + sdar_masks[t].tolist()
                    + [float(rewards[t]), float(rewards_raw[t])]
                    + (components_final[t].tolist() if SAVE_REWARD_COMPONENTS else [])
                    + next_observations[t].tolist()
                    + [float(terminals[t]), float(timeouts[t])]
                )
                writer.writerow(row)
        print(f"Saved CSV: {csv_path}  ({T} rows, {len(header)} cols)")

    result = {
        "observations": observations,
        "actions": actions,
        "actions_raw": actions_raw,
        "prev_actions": sdar_prev_actions,
        "selection_masks": sdar_masks,
        "action_mixes": sdar_mixes,
        "rewards": rewards,
        "rewards_raw": rewards_raw,
        "reward_components": components_final,
        "reward_components_raw": components_raw,
        "reward_component_names": comp_names,
        "reward_component_scales": comp_scales,
        "reward_component_importance": importance_vec,
        "next_observations": next_observations,
        "terminals": terminals,
        "timeouts": timeouts,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "act_mean": act_mean,
        "act_std": act_std,
        "light_ranges_per_zone": light_ranges,
        "expert_actions": expert_actions,
        "executed_actions": executed_actions,
        "exploration_masks": exploration_masks,
        "sdar_forced_repeat": sdar_forced_repeat,
        "glazing_explored": glazing_explored,
        "lighting_explored": lighting_explored,
        "thermal_mode_ids": thermal_mode_ids,
        "lighting_dimming": lighting_dimming,
        "lighting_power_from_action_w": lighting_power_from_action_w,
        "lighting_energy_kwh": lighting_energy_kwh,
    }

    if "expert_id" in temporal_data:
        result["expert_ids"] = np.asarray(temporal_data["expert_id"], dtype=np.int64)[:T]

    return result


if __name__ == "__main__":
    dataset = build_offline_dataset(save_npz=True, save_csv=True)