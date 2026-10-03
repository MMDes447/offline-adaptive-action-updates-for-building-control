"""
Fixed-rate SDAR agent-rollout dataset builder
=============================================

Runs AFTER ``sdar_ablation_evaluation_callback.py`` has evaluated the trained
SDAR-IQL agent in ``SELECTOR_MODE = "constant"`` or ``"periodic"`` and filled
the in-memory logs.
It reconstructs the same offline-RL dataset format from the agent's trajectory so
you can score the policy (return, per-component reward, thermal violations,
switching, energy) and, if you want, re-pool the agent rollout with the training
data.

WHAT'S DIFFERENT FROM THE v8 EXPERT BUILDER
-------------------------------------------
The reward math, observation layout, and diagnostics are IDENTICAL (reward is a
property of states/actions, not of which policy produced them). Only the action
source changes:

  expert builder                     agent builder (this file)
  ------------------------------     --------------------------------------
  action_vector_data["action"]   ->  agent_data["executed_action_norm"]
  action_vector_data["...mask"]  ->  agent_data["selection_mask"]
  prev_action / action_mix        ->  logged/reconstructed SDAR context
  exploration_data (expert vs        (removed — no exploration during eval;
    executed, thermal modes, ...)     selector sampling is logged directly)

Required in-memory globals (filled by the eval callback rollout):
    ZONES_MID, AVAILABLE_GLAZING_STATES
    temperature_data, lighting_data, solar_data, ext_irr_data, wpi_data
    meter_data, output_variable_data, temporal_data
    glazing_state_data, lighting_power_data,
    heating_setpoint_data, cooling_setpoint_data
    agent_data  (observations, active and learned selector probabilities,
                 target rates, policy/executed/physical-change masks,
                 proposal/previous/policy/final actions, and safeguards)

FIXED-RATE SDAR ATTRIBUTION
---------------------------
``selection_mask`` is either the state-independent Bernoulli mask (constant
mode) or deterministic fractional-period mask (periodic mode). It remains the
mask used by the switching-penalty reward, keeping reward accounting comparable
with training. ``executed_selection_mask`` additionally records
safety/deadband overrides. ``actual_change_mask`` records whether the physical
command really changed; it is saved for actuator/switching analysis but does
not silently alter the reward definition.

The fixed rates must match the rates used by the callback. The builder verifies
the logged target-rate vector and refuses to mix constant and periodic logs.

REWARD COMPARABILITY (important)
--------------------------------
This version REQUIRES the pooled decomposed training NPZ as a reward reference.
It loads the exact:

    reward_component_names
    reward_component_scales
    reward_component_importance
    reward_component_clips
    reward_transform

used by training. For the current decomposed run that means thermal importance
3.0, thermal clip 50.0, and clip 10.0 for all other components. The evaluated
rollout never defines its own normalization scales, so rewards remain comparable
across checkpoints and against the experts.

`rewards_raw` is kept as the training-compatible name for the balanced,
post-normalization/post-importance/post-component-clip sum before symlog.
`reward_physical_raw` is additionally saved for the sum of physical W_* reward
components before normalization.
"""

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


# ============================================================
# ROLLOUT LOGS — provided by the eval callback
# ============================================================
# If this file is imported standalone, bind the logging globals from the
# callback module. Python imports the same mutable dictionaries used by the
# callback when both modules run in the same process/kernel.
try:
    agent_data  # noqa: F821  (present when run in the rollout namespace)
except NameError:
    from sdar_ablation_evaluation_callback import (  # noqa: F401
        ZONES_MID,
        AVAILABLE_GLAZING_STATES,
        temperature_data,
        lighting_data,
        solar_data,
        ext_irr_data,
        wpi_data,
        meter_data,
        output_variable_data,
        temporal_data,
        glazing_state_data,
        lighting_power_data,
        heating_setpoint_data,
        cooling_setpoint_data,
        agent_data,
        SELECTOR_MODE,
        PROPOSAL_DETERMINISTIC,
        SELECTOR_SEED,
        PERIODIC_PHASE_SEED,
        CONSTANT_UPDATE_RATES,
    )


# ============================================================
# CONFIG
# ============================================================

# These values are validated against the reference NPZ. They are not used as an
# independent reward definition.
REWARD_TRANSFORM = "symlog"     # "symlog" | "clip" | "none"
REWARD_CLIP_LIMIT = 50.0
SAVE_REWARD_COMPONENTS = True

REWARD_NORMALIZE = True
REWARD_COMPONENT_CLIP = 10.0  # legacy/default clip; per-component clips come from NPZ

REWARD_IMPORTANCE = {
    "energy_lighting":       0.40,
    "hvac_demand_rate":      1.00,
    "thermal_comfort":       3.00,
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

STEPS_PER_HOUR = 6
TIMESTEP_HOURS = 1.0 / STEPS_PER_HOUR

# ---------------------------------------------------------------------------
# EDIT THESE FOR EACH CHECKPOINT EVALUATION.
# Use the decomposed pooled dataset that trained the agent. If it is elsewhere,
# provide the absolute path.
# ---------------------------------------------------------------------------
REFERENCE_REWARD_DATASET = (
    "pooled_sdar_experts_thermal3_tclip50_decomposed.npz"
)
EVAL_CHECKPOINT_PATH = "checkpoints/sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36/sdar_iql_sdar_iql_v10_thermal3_tclip50_14episodes_gru2_seq36_checkpoint_epoch100.pt"   # e.g. "/.../checkpoint_epoch150.pt"
EVAL_CHECKPOINT_EPOCH = None  # optional; inferred from the checkpoint filename
# Metadata only. Set this only if an external evaluation/environment seed is
# actually used; selector and periodic scheduling have their own seeds below.
EVAL_SEED = None
# EVAL_SELECTOR_MODE = str(globals().get("SELECTOR_MODE", "constant")).lower()
SELECTOR_MODE = "periodic"
EVAL_PROPOSAL_DETERMINISTIC = bool(
    globals().get("PROPOSAL_DETERMINISTIC", True)
)
EVAL_SELECTOR_SEED = int(globals().get("SELECTOR_SEED", 20260728))
EVAL_PERIODIC_PHASE_SEED = int(
    globals().get("PERIODIC_PHASE_SEED", EVAL_SELECTOR_SEED)
)
EVAL_FIXED_UPDATE_RATES = dict(
    globals().get(
        "CONSTANT_UPDATE_RATES",
        {
            "glazing": 0.02777,
            "lighting": 0.12440,
            "heating": 0.26803,
            "cooling": 0.26908,
        },
    )
)
EVAL_SAFEGUARD_ENABLED = True
EVAL_WEATHER_FILE = None


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

LIGHTING_INSTALLED_W = {
    "Perimeter_mid_ZN_1": 2231.0,
    "Perimeter_mid_ZN_2": 2231.0,
    "Perimeter_mid_ZN_3": 1412.0,
    "Perimeter_mid_ZN_4": 1412.0,
    "Core_mid":          10586.0,
}

HEATING_SETPOINT_RANGE = (16.0, 24.0)
COOLING_SETPOINT_RANGE = (22.0, 28.0)
ACTION_VALIDATION_ATOL = 0.5


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
    """Physical (actuated) action, read from the eval callback's logs."""
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
# LIGHTING ENERGY FROM THE AGENT'S NORMALIZED ACTION
# ============================================================
# The per-zone normalized lighting action a in [-1, 1] maps to a dimming duty
#   u = (a + 1) / 2 in [0, 1], and E = sum_z nominal_W[z]*u[z] * dt / 1000 (kWh).
# Meter-free: uses the executed action, not the logged Lights Electricity Rate.

def compute_lighting_energy_kwh(action_norm_t):
    a = np.asarray(action_norm_t, dtype=np.float64)
    dimming = np.clip((a[LIGHT_SLICE] + 1.0) / 2.0, 0.0, 1.0)
    nominal_w = np.array([LIGHTING_INSTALLED_W[zone] for zone in ZONES_MID], dtype=np.float64)
    total_power_w = float((nominal_w * dimming).sum())
    return total_power_w * TIMESTEP_HOURS / 1000.0


# ============================================================
# REWARD — components + scalar (identical logic to the v8 builder)
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


def compute_reward_components(t, action_norm_t, selection_mask_t):
    occupied_next = bool(temporal_data["is_occupied"][t + 1])
    occupied_lighting = bool(temporal_data["is_occupied"][t])

    hvac_demand_w = output_variable_data["HVAC_electricity_demand_rate"][t + 1]
    r_hvac_demand = -W_HVAC_DEMAND * (hvac_demand_w / 1000.0)

    lighting_energy_kwh = compute_lighting_energy_kwh(action_norm_t)
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
        b = np.asarray(selection_mask_t)
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


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _npz_scalar(data, key, default=None):
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"Expected scalar NPZ field {key!r}, got shape {value.shape}.")
    return value.reshape(()).item()


def load_reward_reference(reference_dataset_path, comp_names):
    """
    Load the immutable reward ruler from the pooled decomposed training NPZ.

    There is deliberately no rollout-local fallback. A missing or incompatible
    reference dataset is an error because otherwise checkpoint rewards would be
    calculated on different scales.
    """
    path = Path(reference_dataset_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            "Reward reference dataset not found: "
            f"{path}. Set REFERENCE_REWARD_DATASET to the exact pooled NPZ used "
            "to train the decomposed agent."
        )

    required = {
        "reward_component_names",
        "reward_component_scales",
        "reward_component_importance",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(
                f"Reference NPZ {path} is missing required reward metadata: {missing}"
            )

        ref_names = [str(x) for x in np.asarray(data["reward_component_names"]).tolist()]
        if ref_names != list(comp_names):
            raise ValueError(
                "Reward component order differs from the training dataset.\n"
                f"  builder:   {list(comp_names)}\n"
                f"  reference: {ref_names}"
            )

        scales = np.asarray(data["reward_component_scales"], dtype=np.float64)
        importance = np.asarray(
            data["reward_component_importance"], dtype=np.float64
        )
        if scales.shape != (len(comp_names),):
            raise ValueError(
                f"reward_component_scales shape {scales.shape}; "
                f"expected {(len(comp_names),)}"
            )
        if importance.shape != (len(comp_names),):
            raise ValueError(
                f"reward_component_importance shape {importance.shape}; "
                f"expected {(len(comp_names),)}"
            )
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("Reference reward scales must all be finite and positive.")
        if np.any(~np.isfinite(importance)):
            raise ValueError("Reference reward importance values must all be finite.")

        if "reward_component_clips" in data.files:
            clips = np.asarray(data["reward_component_clips"], dtype=np.float64)
        else:
            legacy_clip = float(
                _npz_scalar(data, "reward_component_clip", REWARD_COMPONENT_CLIP)
            )
            clips = np.full(len(comp_names), legacy_clip, dtype=np.float64)
            if "thermal_reward_component_clip" in data.files:
                thermal_i = comp_names.index("thermal_comfort")
                clips[thermal_i] = float(
                    _npz_scalar(data, "thermal_reward_component_clip")
                )

        if clips.shape != (len(comp_names),):
            raise ValueError(
                f"reward_component_clips shape {clips.shape}; "
                f"expected {(len(comp_names),)}"
            )
        # A negative or non-finite stored clip means unbounded for that component.
        clips = np.where(np.isfinite(clips) & (clips >= 0.0), clips, np.inf)

        transform = str(
            _npz_scalar(data, "reward_transform", REWARD_TRANSFORM)
        )
        reweighting_tag = str(
            _npz_scalar(data, "reward_reweighting_tag", "")
        )
        reference_mask_value = float(
            _npz_scalar(data, "action_mask_value", ACTION_MASK_VALUE)
        )

    configured_importance = np.array(
        [REWARD_IMPORTANCE.get(name, 1.0) for name in comp_names],
        dtype=np.float64,
    )
    if not np.allclose(configured_importance, importance, atol=1e-7, rtol=0.0):
        raise ValueError(
            "Builder REWARD_IMPORTANCE differs from the training NPZ. "
            "Do not evaluate until the reward definitions match.\n"
            f"  builder:   {configured_importance.tolist()}\n"
            f"  reference: {importance.tolist()}"
        )
    if transform != REWARD_TRANSFORM:
        raise ValueError(
            f"Builder transform {REWARD_TRANSFORM!r} differs from reference "
            f"transform {transform!r}."
        )
    if not np.isclose(reference_mask_value, ACTION_MASK_VALUE):
        raise ValueError(
            f"Reference action_mask_value={reference_mask_value} differs from "
            f"builder ACTION_MASK_VALUE={ACTION_MASK_VALUE}."
        )

    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "names": list(comp_names),
        "scales": scales,
        "importance": importance,
        "clips": clips,
        "transform": transform,
        "reweighting_tag": reweighting_tag,
    }


def apply_reward_reference(components_raw, reward_reference):
    """Apply training scales, importance, and per-component clips exactly."""
    scales = reward_reference["scales"]
    importance = reward_reference["importance"]
    clips = reward_reference["clips"]

    unbounded = (
        components_raw.astype(np.float64)
        / scales[None, :]
        * importance[None, :]
    )
    components_final = np.clip(
        unbounded,
        -clips[None, :],
        clips[None, :],
    )
    return components_final.astype(np.float32), unbounded.astype(np.float32)


def _infer_checkpoint_epoch(checkpoint_path, checkpoint_epoch):
    if checkpoint_epoch is not None:
        epoch = int(checkpoint_epoch)
        if epoch < 0:
            raise ValueError("checkpoint_epoch must be non-negative.")
        return epoch
    if checkpoint_path:
        match = re.search(r"epoch[_-]?(\d+)", Path(checkpoint_path).name, re.IGNORECASE)
        if match:
            return int(match.group(1))
    raise ValueError(
        "Checkpoint epoch is required for unambiguous output naming. Set "
        "EVAL_CHECKPOINT_EPOCH or use a checkpoint filename containing "
        "'epoch<number>'."
    )


def _checkpoint_provenance(checkpoint_path, checkpoint_epoch):
    epoch = _infer_checkpoint_epoch(checkpoint_path, checkpoint_epoch)
    if checkpoint_path is None:
        return {
            "epoch": epoch,
            "path": "",
            "sha256": "",
        }

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return {
        "epoch": epoch,
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
    }


# ============================================================
# DIAGNOSTICS (identical to the v8 builder)
# ============================================================

def _print_reward_diagnostics(
    reward_physical_raw,
    reward_balanced,
    rewards_tx,
    components_arr,
):
    print("\n=== Reward diagnostics ===")
    qs = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
    print(
        f"  physical raw: mean={reward_physical_raw.mean():+.4f}  "
        f"std={reward_physical_raw.std():.4f}"
    )
    print(
        f"                min={reward_physical_raw.min():+.4f}    "
        f"max={reward_physical_raw.max():+.4f}"
    )
    print(f"               quantiles {qs} = "
          f"{[round(float(q), 3) for q in np.quantile(reward_physical_raw, qs)]}")
    print(
        f"  balanced:     mean={reward_balanced.mean():+.4f}  "
        f"std={reward_balanced.std():.4f}"
    )
    print(
        f"                min={reward_balanced.min():+.4f}    "
        f"max={reward_balanced.max():+.4f}"
    )
    print(f"               quantiles {qs} = "
          f"{[round(float(q), 3) for q in np.quantile(reward_balanced, qs)]}")
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
    err = comp_sum - reward_balanced
    print("\n=== Component sum check ===")
    print(f"  mean abs(sum_components - balanced): {np.mean(np.abs(err)):.8f}")
    print(f"  max  abs(sum_components - balanced): {np.max(np.abs(err)):.8f}")


def _print_normalization(
    comp_names,
    comp_scales,
    importance_vec,
    component_clips,
    scales_source,
):
    print("\n=== Reward component normalization ===")
    print(f"  REWARD_NORMALIZE={REWARD_NORMALIZE}  scales={scales_source}")
    print(f"  {'component':24s}  {'scale(mean|abs|)':>16s}  {'importance':>10s}  "
          f"{'clip':>8s}  {'eff_weight':>10s}")
    for nm, sc, imp, clip in zip(
        comp_names, comp_scales, importance_vec, component_clips
    ):
        eff = imp / sc if sc != 0 else 0.0
        clip_name = "none" if not np.isfinite(clip) else f"{clip:.1f}"
        print(
            f"  {nm:24s}  {sc:16.6f}  {imp:10.3f}  "
            f"{clip_name:>8s}  {eff:10.4f}"
        )


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
    print("\n=== SDAR selection mask diagnostics (agent selector) ===")
    print("  b=1 means proposal selected, b=0 means previous command repeated")
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


def _print_selector_probability_diagnostics(
    selection_probabilities,
    learned_selection_probabilities,
    selection_masks,
    executed_selection_masks,
    actual_change_masks,
    target_update_rates,
    force_first_update,
    selector_mode,
):
    # Exclude every explicitly forced step when checking the selector
    # realization. Normally this removes only transition zero.
    keep = ~np.asarray(force_first_update, dtype=bool)
    if not np.any(keep):
        raise ValueError("No unforced transitions are available for diagnostics.")

    probabilities = selection_probabilities[keep]
    learned_probabilities = learned_selection_probabilities[keep]
    masks = selection_masks[keep]
    executed_masks = executed_selection_masks[keep]
    changed_masks = actual_change_masks[keep]
    target_rates = target_update_rates[keep]

    groups = {
        "glazing": GLAZE_SLICE,
        "lighting": LIGHT_SLICE,
        "heating": HEAT_SLICE,
        "cooling": COOL_SLICE,
        "overall": slice(0, ACTION_DIM),
    }

    print(
        "\n=== Selector probability diagnostics "
        f"(mode={selector_mode}) ==="
    )
    stats = {}
    for name, slc in groups.items():
        p = probabilities[:, slc]
        lp = learned_probabilities[:, slc]
        m = masks[:, slc]
        em = executed_masks[:, slc]
        cm = changed_masks[:, slc]
        tr = target_rates[:, slc]
        mean_probability = float(p.mean())
        learned_mean_probability = float(lp.mean())
        target_rate = float(tr.mean())
        selected_rate = float(m.mean())
        executed_update_rate = float(em.mean())
        actual_change_rate = float(cm.mean())
        difference = selected_rate - target_rate
        stats[name] = {
            "mean_probability": mean_probability,
            "learned_mean_probability": learned_mean_probability,
            "target_rate": target_rate,
            "sampled_rate": selected_rate,
            "selected_rate": selected_rate,
            "executed_update_rate": executed_update_rate,
            "actual_change_rate": actual_change_rate,
            "difference": difference,
        }
        print(
            f"  {name:10s} "
            f"target={target_rate:.4f}  "
            f"selected={selected_rate:.4f}  "
            f"executed={executed_update_rate:.4f}  "
            f"changed={actual_change_rate:.4f}  "
            f"learned_p={learned_mean_probability:.4f}  "
            f"selected-target={difference:+.4f}"
        )

    # Threshold agreement is meaningful only for threshold mode. It is NaN for
    # fixed-rate modes so nobody mistakes their masks for learned decisions.
    if selector_mode == "threshold":
        threshold_masks = (probabilities >= 0.5).astype(np.float32)
        stats["threshold_agreement"] = float(
            np.mean(masks == threshold_masks)
        )
    else:
        stats["threshold_agreement"] = None
    return stats


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
    if TIMEOUT_STEPS is None or TIMEOUT_STEPS <= 0:
        return timeouts
    for i in range(T):
        if (i + 1) % int(TIMEOUT_STEPS) == 0:
            timeouts[i] = 1.0
    return timeouts


def _denormalize_linear(values, physical_range):
    low, high = physical_range
    values = np.asarray(values, dtype=np.float64)
    return low + 0.5 * (values + 1.0) * (high - low)


def _validate_actuated_actions(actions, actions_raw):
    """Verify all 19 normalized executed actions against physical actuation logs."""
    n_glazing = len(AVAILABLE_GLAZING_STATES)
    glz_ids_from_norm = np.clip(
        np.round((actions[:, GLAZE_SLICE] + 1.0) * 0.5 * (n_glazing - 1)),
        0,
        n_glazing - 1,
    ).astype(np.int64)
    glz_ids_logged = actions_raw[:, GLAZE_SLICE].astype(np.int64)
    if not np.array_equal(glz_ids_from_norm, glz_ids_logged):
        mismatch = np.argwhere(glz_ids_from_norm != glz_ids_logged)[0]
        t, dim = (int(mismatch[0]), int(mismatch[1]))
        raise ValueError(
            "Glazing decoded from executed_action_norm does not match the "
            f"actuation log at transition {t}, glazing dim {dim}: "
            f"decoded={glz_ids_from_norm[t, dim]}, logged={glz_ids_logged[t, dim]}."
        )

    nominal_power_w = np.array(
        [LIGHTING_INSTALLED_W[zone] for zone in ZONES_MID],
        dtype=np.float64,
    )
    light_decoded = (
        np.clip((actions[:, LIGHT_SLICE] + 1.0) / 2.0, 0.0, 1.0)
        * nominal_power_w[None, :]
    )
    heat_decoded = _denormalize_linear(
        actions[:, HEAT_SLICE], HEATING_SETPOINT_RANGE
    )
    cool_decoded = _denormalize_linear(
        actions[:, COOL_SLICE], COOLING_SETPOINT_RANGE
    )

    checks = [
        (
            "lighting",
            light_decoded,
            actions_raw[:, LIGHT_SLICE],
            "W",
        ),
        (
            "heating setpoint",
            heat_decoded,
            actions_raw[:, HEAT_SLICE],
            "degC",
        ),
        (
            "cooling setpoint",
            cool_decoded,
            actions_raw[:, COOL_SLICE],
            "degC",
        ),
    ]
    for label, decoded, logged, unit in checks:
        if not np.allclose(
            decoded,
            logged,
            atol=ACTION_VALIDATION_ATOL,
            rtol=1e-4,
        ):
            abs_error = np.abs(decoded - logged)
            t, dim = np.unravel_index(np.argmax(abs_error), abs_error.shape)
            raise ValueError(
                f"{label} decoded from executed_action_norm does not match the "
                f"actuation log. Largest error at transition {t}, local dim {dim}: "
                f"decoded={decoded[t, dim]:.6g} {unit}, "
                f"logged={logged[t, dim]:.6g} {unit}, "
                f"abs_error={abs_error[t, dim]:.6g} {unit}."
            )


def _reconstruct_sdar_context(actions, selection_masks):
    previous_keys = (
        "prev_action_norm_before",
        "previous_action_norm",
        "prev_action_norm",
        "previous_executed_action_norm",
    )
    previous_key = next((k for k in previous_keys if k in agent_data), None)

    if previous_key is not None:
        logged_prev = np.asarray(agent_data[previous_key], dtype=np.float32)
        if logged_prev.ndim != 2 or logged_prev.shape[1] != ACTION_DIM:
            raise ValueError(
                f"agent_data[{previous_key!r}] has shape {logged_prev.shape}; "
                f"expected (*, {ACTION_DIM})."
            )
        if logged_prev.shape[0] < len(actions):
            raise ValueError(
                f"agent_data[{previous_key!r}] has only {logged_prev.shape[0]} "
                f"rows for {len(actions)} transitions."
            )
        prev_actions = logged_prev[:len(actions)].copy()
    else:
        if not np.all(selection_masks[0] == 1.0):
            raise ValueError(
                "The first selection mask is not a full update, but the callback "
                "did not log previous_action_norm. The initial SDAR context cannot "
                "be reconstructed without leaking action[0] into prev_action[0]."
            )
        prev_actions = actions.copy()
        prev_actions[1:] = actions[:-1]

    if len(actions) > 1 and not np.allclose(
        prev_actions[1:],
        actions[:-1],
        atol=1e-5,
        rtol=0.0,
    ):
        max_error = float(
            np.max(np.abs(prev_actions[1:] - actions[:-1]))
        )
        raise ValueError(
            "Logged/reconstructed previous actions do not equal the preceding "
            f"executed actions (max abs error {max_error:.6g})."
        )

    action_mixes = (
        (1.0 - selection_masks) * prev_actions
        + selection_masks * ACTION_MASK_VALUE
    ).astype(np.float32)

    for key in ("action_mix", "action_mixes"):
        if key not in agent_data:
            continue
        logged_mix = np.asarray(agent_data[key], dtype=np.float32)
        if logged_mix.ndim != 2 or logged_mix.shape[1] != ACTION_DIM:
            raise ValueError(
                f"agent_data[{key!r}] has shape {logged_mix.shape}; "
                f"expected (*, {ACTION_DIM})."
            )
        if logged_mix.shape[0] < len(actions):
            raise ValueError(
                f"agent_data[{key!r}] has only {logged_mix.shape[0]} rows for "
                f"{len(actions)} transitions."
            )
        if not np.allclose(
            action_mixes,
            logged_mix[:len(actions)],
            atol=1e-5,
            rtol=0.0,
        ):
            max_error = float(
                np.max(np.abs(action_mixes - logged_mix[:len(actions)]))
            )
            raise ValueError(
                f"Reconstructed action_mix differs from agent_data[{key!r}] "
                f"(max abs error {max_error:.6g})."
            )
        break

    return prev_actions.astype(np.float32), action_mixes


def _logged_matrix(key, n_rows, n_cols=ACTION_DIM, dtype=np.float32):
    """Load a required per-step matrix from ``agent_data`` with strict shape."""
    if key not in agent_data:
        raise KeyError(
            f"agent_data is missing {key!r}. Run the rollout with "
            "sdar_ablation_evaluation_callback.py before building the dataset."
        )
    value = np.asarray(agent_data[key], dtype=dtype)
    expected = (n_rows, n_cols)
    if value.shape != expected:
        raise ValueError(
            f"agent_data[{key!r}] has shape {value.shape}; expected {expected}."
        )
    return value


def _logged_vector(key, n_rows, dtype=None):
    """Load a required per-step vector from ``agent_data`` with strict shape."""
    if key not in agent_data:
        raise KeyError(
            f"agent_data is missing {key!r}. Run the rollout with "
            "sdar_ablation_evaluation_callback.py before building the dataset."
        )
    value = np.asarray(agent_data[key], dtype=dtype)
    if value.shape != (n_rows,):
        raise ValueError(
            f"agent_data[{key!r}] has shape {value.shape}; expected ({n_rows},)."
        )
    return value


def _validate_binary_matrix(name, value):
    """Return a float32 binary matrix after checking finite near-binary input."""
    if np.any(~np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    rounded = np.rint(value)
    if not np.allclose(value, rounded, atol=1e-6, rtol=0.0):
        bad = np.argwhere(np.abs(value - rounded) > 1e-6)[0]
        t, dim = int(bad[0]), int(bad[1])
        raise ValueError(
            f"{name} must be binary; found {value[t, dim]} at step {t}, dim {dim}."
        )
    return rounded.astype(np.float32)


def _fixed_rate_vector(rates):
    """Expand per-subsystem fixed rates into the 19-dimensional action layout."""
    required = ("glazing", "lighting", "heating", "cooling")
    missing = [name for name in required if name not in rates]
    extra = [name for name in rates if name not in required]
    if missing or extra:
        raise ValueError(
            "fixed_update_rates must contain exactly glazing, lighting, "
            f"heating, and cooling; missing={missing}, extra={extra}."
        )
    clean = {name: float(rates[name]) for name in required}
    bad = {name: rate for name, rate in clean.items() if not 0.0 <= rate <= 1.0}
    if bad:
        raise ValueError(f"Fixed update rates must lie in [0,1]; got {bad}.")
    return clean, np.array(
        [clean["glazing"]] * 4
        + [clean["lighting"]] * 5
        + [clean["heating"]] * 5
        + [clean["cooling"]] * 5,
        dtype=np.float32,
    )


def _reconstruct_periodic_masks(n_steps, rate_vector, phase_seed):
    """Reproduce the callback's deterministic fractional-period schedule."""
    if n_steps < 1:
        raise ValueError("n_steps must be at least one.")
    rate_vector = np.asarray(rate_vector, dtype=np.float32)
    if rate_vector.shape != (ACTION_DIM,):
        raise ValueError(
            f"rate_vector has shape {rate_vector.shape}; expected ({ACTION_DIM},)."
        )

    phase_rng = np.random.default_rng(int(phase_seed))
    accumulator = phase_rng.uniform(
        0.0, 1.0, size=(ACTION_DIM,)
    ).astype(np.float32)
    masks = np.zeros((n_steps, ACTION_DIM), dtype=np.float32)

    # The evaluation callback explicitly forces the first decision and does not
    # advance the periodic accumulator on that decision.
    masks[0] = 1.0
    for t in range(1, n_steps):
        accumulator = accumulator + rate_vector
        mask = (accumulator >= 1.0).astype(np.float32)
        masks[t] = mask
        accumulator = accumulator - mask
    return masks


# ============================================================
# MAIN BUILDER
# ============================================================

def build_agent_dataset(
    save_npz=True,
    save_csv=True,
    prefix=None,
    reference_dataset_path=REFERENCE_REWARD_DATASET,
    checkpoint_path=EVAL_CHECKPOINT_PATH,
    checkpoint_epoch=EVAL_CHECKPOINT_EPOCH,
    eval_seed=EVAL_SEED,
    selector_mode=EVAL_SELECTOR_MODE,
    proposal_deterministic=EVAL_PROPOSAL_DETERMINISTIC,
    selector_seed=EVAL_SELECTOR_SEED,
    fixed_update_rates=EVAL_FIXED_UPDATE_RATES,
    periodic_phase_seed=EVAL_PERIODIC_PHASE_SEED,
    safeguard_enabled=EVAL_SAFEGUARD_ENABLED,
    weather_file=EVAL_WEATHER_FILE,
):
    selector_mode = str(selector_mode).lower()
    if selector_mode not in {"constant", "periodic"}:
        raise ValueError(
            "This builder supports only the fixed-rate SDAR modes 'constant' "
            f"and 'periodic'; got {selector_mode!r}."
        )
    proposal_deterministic = bool(proposal_deterministic)
    if not proposal_deterministic:
        raise ValueError(
            "The SDAR ablation keeps the proposal deterministic in every mode."
        )
    selector_seed = int(selector_seed)
    periodic_phase_seed = int(periodic_phase_seed)
    fixed_update_rates, expected_rate_vector = _fixed_rate_vector(
        fixed_update_rates
    )
    policy_tag = f"sdar_iql_{selector_mode}_ablation_agent"

    comp_names = _component_names()
    reward_reference = load_reward_reference(
        reference_dataset_path,
        comp_names,
    )
    checkpoint = _checkpoint_provenance(
        checkpoint_path,
        checkpoint_epoch,
    )
    if prefix is None:
        proposal_tag = (
            "propdet" if proposal_deterministic else "propstoch"
        )
        schedule_seed_tag = (
            f"s{selector_seed}"
            if selector_mode == "constant"
            else f"phase{periodic_phase_seed}"
        )
        prefix = (
            f"agent_eval_dataset_sdar_ablation_{selector_mode}"
            f"_epoch{checkpoint['epoch']}"
            "_thermal3_tclip50"
            f"_{proposal_tag}"
            f"_{schedule_seed_tag}"
        )
    prefix = str(Path(prefix).expanduser())
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)

    print("=== Evaluation provenance ===")
    print(f"  checkpoint epoch: {checkpoint['epoch']}")
    print(f"  checkpoint path:  {checkpoint['path'] or '(not supplied)'}")
    print(f"  reference NPZ:    {reward_reference['path']}")
    print(
        "  reward tag:       "
        f"{reward_reference['reweighting_tag'] or '(not stored)'}"
    )
    print(f"  selector mode:    {selector_mode}")
    print(
        "  proposal deterministic: "
        f"{proposal_deterministic}"
    )
    print(f"  selector seed:    {selector_seed}")
    if selector_mode == "periodic":
        print(f"  periodic phase seed: {periodic_phase_seed}")
    print(f"  fixed rates:      {fixed_update_rates}")
    print(f"  safeguard:        {bool(safeguard_enabled)}")
    print(f"  eval seed:        {eval_seed}")
    print(f"  weather file:     {weather_file}")

    n = len(temperature_data[ZONES_MID[0]])
    T = n - 1
    if T < 1:
        raise ValueError(f"Not enough logged steps to build transitions (n={n}).")

    # ---- length sanity check (agent logs only) ----
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
        "agent_observation": len(agent_data["observation"]),
        "agent_selection_logits": len(agent_data["selection_logits"]),
        "agent_selection_probability": len(
            agent_data["selection_probability"]
        ),
        "agent_learned_selection_logits": len(
            agent_data["learned_selection_logits"]
        ),
        "agent_learned_selection_probability": len(
            agent_data["learned_selection_probability"]
        ),
        "agent_selection_mask": len(agent_data["selection_mask"]),
        "agent_executed_selection_mask": len(
            agent_data["executed_selection_mask"]
        ),
        "agent_actual_change_mask": len(agent_data["actual_change_mask"]),
        "agent_target_update_rate": len(agent_data["target_update_rate"]),
        "agent_proposal_action_norm": len(agent_data["proposal_action_norm"]),
        "agent_prev_action_norm_before": len(
            agent_data["prev_action_norm_before"]
        ),
        "agent_policy_executed_action_norm": len(
            agent_data["policy_executed_action_norm"]
        ),
        "agent_executed_action_norm": len(agent_data["executed_action_norm"]),
        "agent_override_dims": len(agent_data["override_dims"]),
        "agent_deadband_applied": len(agent_data["deadband_applied"]),
        "agent_lighting_safety_active": len(
            agent_data["lighting_safety_active"]
        ),
        "agent_lighting_safety_override": len(
            agent_data["lighting_safety_override"]
        ),
        "agent_selector_mode": len(agent_data["selector_mode"]),
        "agent_selector_stochastic": len(agent_data["selector_stochastic"]),
        "agent_force_first_update": len(agent_data["force_first_update"]),
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

    # ---- observations ----
    observations = np.array([build_observation(t) for t in range(T)], dtype=np.float32)
    next_observations = np.array([build_observation(t + 1) for t in range(T)], dtype=np.float32)

    obs_dim = len(obs_column_names())
    if observations.shape[1] != obs_dim:
        raise ValueError(f"Expected {obs_dim}-dim observations, got {observations.shape}")

    # Integrity: the obs we rebuild here must equal what the agent actually saw.
    agent_obs = np.asarray(agent_data["observation"], dtype=np.float32)[:T]
    if not np.allclose(observations, agent_obs, atol=1e-4):
        max_err = float(np.max(np.abs(observations - agent_obs)))
        raise ValueError(
            "Rebuilt observations differ from what the agent saw "
            f"(max abs err {max_err:.6g}). The dataset obs layout has drifted "
            "from the eval callback's build_observation.")

    # ---- complete fixed-rate SDAR decision/execution trace ----
    agent_actions = _logged_matrix("executed_action_norm", n)
    agent_logits = _logged_matrix("selection_logits", n)
    agent_probabilities = _logged_matrix("selection_probability", n)
    agent_learned_logits = _logged_matrix("learned_selection_logits", n)
    agent_learned_probabilities = _logged_matrix(
        "learned_selection_probability", n
    )
    agent_masks = _logged_matrix("selection_mask", n)
    agent_executed_masks = _logged_matrix("executed_selection_mask", n)
    agent_actual_change_masks = _logged_matrix("actual_change_mask", n)
    agent_target_rates = _logged_matrix("target_update_rate", n)
    agent_proposals = _logged_matrix("proposal_action_norm", n)
    agent_prev_actions = _logged_matrix("prev_action_norm_before", n)
    agent_policy_actions = _logged_matrix("policy_executed_action_norm", n)
    agent_override_dims = _logged_matrix("override_dims", n)
    agent_deadband = _logged_matrix("deadband_applied", n, n_cols=5)
    agent_lighting_safety_active = _logged_vector(
        "lighting_safety_active", n, dtype=bool
    )
    agent_lighting_safety_override = _logged_vector(
        "lighting_safety_override", n, dtype=bool
    )
    agent_selector_modes = _logged_vector("selector_mode", n, dtype=str)
    agent_selector_stochastic = _logged_vector(
        "selector_stochastic", n, dtype=bool
    )
    agent_force_first_update = _logged_vector(
        "force_first_update", n, dtype=bool
    )

    actions = agent_actions[:T].copy()
    selection_logits = agent_logits[:T].copy()
    selection_probabilities = agent_probabilities[:T].copy()
    learned_selection_logits = agent_learned_logits[:T].copy()
    learned_selection_probabilities = agent_learned_probabilities[:T].copy()
    selection_masks = _validate_binary_matrix(
        "selection_mask", agent_masks[:T]
    )
    executed_selection_masks = _validate_binary_matrix(
        "executed_selection_mask", agent_executed_masks[:T]
    )
    actual_change_masks = _validate_binary_matrix(
        "actual_change_mask", agent_actual_change_masks[:T]
    )
    target_update_rates = agent_target_rates[:T].copy()
    proposal_actions = agent_proposals[:T].copy()
    logged_prev_actions = agent_prev_actions[:T].copy()
    policy_executed_actions = agent_policy_actions[:T].copy()
    override_dims = _validate_binary_matrix(
        "override_dims", agent_override_dims[:T]
    )
    deadband_applied = _validate_binary_matrix(
        "deadband_applied", agent_deadband[:T]
    )
    lighting_safety_active = agent_lighting_safety_active[:T].copy()
    lighting_safety_override = agent_lighting_safety_override[:T].copy()
    selector_modes = agent_selector_modes[:T].copy()
    selector_stochastic = agent_selector_stochastic[:T].copy()
    force_first_update = agent_force_first_update[:T].copy()

    if not np.all(selector_modes == selector_mode):
        found = sorted(set(selector_modes.tolist()))
        raise ValueError(
            f"The logs are not a pure {selector_mode}-SDAR rollout; "
            f"agent_data['selector_mode'] contains {found}."
        )
    expected_stochastic = selector_mode == "constant"
    if not np.all(selector_stochastic == expected_stochastic):
        raise ValueError(
            f"{selector_mode}-SDAR must log selector_stochastic="
            f"{expected_stochastic} at every step."
        )
    if not force_first_update[0] or np.any(force_first_update[1:]):
        raise ValueError(
            "Expected exactly transition zero to have force_first_update=True."
        )
    if not np.all(selection_masks[0] == 1.0):
        raise ValueError("The forced first selection mask is not all ones.")
    periodic_schedule_verified = False
    if selector_mode == "periodic":
        expected_periodic_masks = _reconstruct_periodic_masks(
            T,
            expected_rate_vector,
            periodic_phase_seed,
        )
        if not np.array_equal(selection_masks, expected_periodic_masks):
            mismatch = np.argwhere(
                selection_masks != expected_periodic_masks
            )[0]
            t, dim = int(mismatch[0]), int(mismatch[1])
            raise ValueError(
                "Logged periodic mask does not match the deterministic "
                f"fractional-period schedule at step {t}, dim {dim}. Check "
                "PERIODIC_PHASE_SEED and fixed update rates."
            )
        periodic_schedule_verified = True
        print("  periodic schedule: verified against phase seed and fixed rates")

    if np.any(~np.isfinite(selection_logits)):
        raise ValueError("selection_logits contains NaN or infinite values.")
    if np.any(~np.isfinite(selection_probabilities)):
        raise ValueError(
            "selection_probability contains NaN or infinite values."
        )
    if (
        np.any(selection_probabilities < -1e-7)
        or np.any(selection_probabilities > 1.0 + 1e-7)
    ):
        raise ValueError(
            "selection_probability must lie in [0,1]: "
            f"min={selection_probabilities.min()}, "
            f"max={selection_probabilities.max()}."
        )

    if np.any(~np.isfinite(learned_selection_logits)):
        raise ValueError("learned_selection_logits contains non-finite values.")
    if np.any(~np.isfinite(learned_selection_probabilities)):
        raise ValueError(
            "learned_selection_probability contains non-finite values."
        )
    if (
        np.any(learned_selection_probabilities < -1e-7)
        or np.any(learned_selection_probabilities > 1.0 + 1e-7)
    ):
        raise ValueError("learned_selection_probability must lie in [0,1].")

    reconstructed_probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(selection_logits, -80.0, 80.0))
    )
    if not np.allclose(
        selection_probabilities,
        reconstructed_probabilities,
        atol=1e-6,
        rtol=1e-6,
    ):
        max_err = float(
            np.max(
                np.abs(
                    selection_probabilities
                    - reconstructed_probabilities
                )
            )
        )
        raise ValueError(
            "selection_probability is inconsistent with sigmoid(logits): "
            f"max abs error={max_err:.6g}."
        )

    learned_reconstructed = 1.0 / (
        1.0 + np.exp(-np.clip(learned_selection_logits, -80.0, 80.0))
    )
    if not np.allclose(
        learned_selection_probabilities,
        learned_reconstructed,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError(
            "learned_selection_probability is inconsistent with "
            "sigmoid(learned_selection_logits)."
        )

    expected_rates = np.broadcast_to(expected_rate_vector, (T, ACTION_DIM))
    if not np.allclose(
        target_update_rates, expected_rates, atol=1e-7, rtol=0.0
    ):
        max_err = float(np.max(np.abs(target_update_rates - expected_rates)))
        raise ValueError(
            "Logged target_update_rate does not match fixed_update_rates "
            f"(max abs error {max_err:.6g})."
        )
    if not np.allclose(
        selection_probabilities, target_update_rates, atol=1e-7, rtol=0.0
    ):
        raise ValueError(
            "In fixed-rate modes, active selection probabilities must equal the "
            "logged target update rates."
        )

    # Normalized executed actions must be finite and lie within [-1, 1].
    if np.any(~np.isfinite(actions)):
        raise ValueError("executed_action_norm contains NaN or infinite values.")
    if np.any(actions < -1.0001) or np.any(actions > 1.0001):
        raise ValueError(
            f"Normalized actions outside [-1,1]: min={actions.min()}, max={actions.max()}")

    for name, value in {
        "proposal_action_norm": proposal_actions,
        "prev_action_norm_before": logged_prev_actions,
        "policy_executed_action_norm": policy_executed_actions,
    }.items():
        if np.any(~np.isfinite(value)):
            raise ValueError(f"{name} contains NaN or infinite values.")
        if np.any(value < -1.0001) or np.any(value > 1.0001):
            raise ValueError(
                f"{name} is outside [-1,1]: min={value.min()}, max={value.max()}."
            )

    expected_policy_actions = (
        (1.0 - selection_masks) * logged_prev_actions
        + selection_masks * proposal_actions
    )
    if not np.allclose(
        policy_executed_actions,
        expected_policy_actions,
        atol=1e-5,
        rtol=0.0,
    ):
        max_err = float(
            np.max(np.abs(policy_executed_actions - expected_policy_actions))
        )
        raise ValueError(
            f"policy_executed_action_norm is inconsistent with the {selector_mode} "
            f"mask/proposal/previous action (max abs error {max_err:.6g})."
        )

    expected_executed_masks = np.clip(
        selection_masks + override_dims, 0.0, 1.0
    )
    if not np.array_equal(executed_selection_masks, expected_executed_masks):
        raise ValueError(
            "executed_selection_mask must equal clip(selection_mask + "
            "override_dims, 0, 1)."
        )
    repeated_error = np.abs(actions - logged_prev_actions) * (
        1.0 - executed_selection_masks
    )
    if float(repeated_error.max()) > 1e-5:
        raise ValueError(
            "A dimension marked as repeated differs from its previous action; "
            f"max error={float(repeated_error.max()):.6g}."
        )
    if np.any(actual_change_masks > executed_selection_masks):
        raise ValueError(
            "actual_change_mask contains a change where executed_selection_mask "
            "says the previous command was repeated."
        )

    # Validate all physical actuations against the executed normalized action.
    actions_raw = np.array([build_action_raw(t) for t in range(T)], dtype=np.float32)
    _validate_actuated_actions(actions, actions_raw)

    # ---- reconstruct SDAR prev_action / action_mix from the agent trajectory ----
    prev_actions, action_mixes = _reconstruct_sdar_context(
        actions,
        selection_masks,
    )
    if not np.allclose(
        prev_actions, logged_prev_actions, atol=1e-6, rtol=0.0
    ):
        raise ValueError(
            "The SDAR previous-action context differs from "
            "prev_action_norm_before."
        )

    # ---- physical lighting quantities from the executed dimming ----
    lighting_dimming = np.clip((actions[:, LIGHT_SLICE] + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
    nominal_power_w = np.array([LIGHTING_INSTALLED_W[zone] for zone in ZONES_MID], dtype=np.float32)
    lighting_power_from_action_w = (lighting_dimming * nominal_power_w[None, :]).astype(np.float32)
    lighting_energy_kwh = (
        lighting_power_from_action_w.sum(axis=1) * TIMESTEP_HOURS / 1000.0).astype(np.float32)

    # ---- rewards ----
    components_raw = np.zeros((T, len(comp_names)), dtype=np.float32)
    for t in range(T):
        comps = compute_reward_components(t, actions[t], selection_masks[t])
        for i, name in enumerate(comp_names):
            components_raw[t, i] = comps[name]

    components_final, components_unbounded = apply_reward_reference(
        components_raw,
        reward_reference,
    )
    comp_scales = reward_reference["scales"]
    importance_vec = reward_reference["importance"]
    component_clips = reward_reference["clips"]
    scales_source = f"training reference: {reward_reference['path']}"

    reward_physical_raw = components_raw.sum(axis=1).astype(np.float32)
    reward_balanced = components_final.sum(axis=1).astype(np.float32)
    # Legacy training-compatible key: `rewards_raw` means balanced pre-symlog.
    rewards_raw = reward_balanced
    rewards_tx = _transform_vec(reward_balanced)
    rewards = rewards_tx.astype(np.float32)

    terminals = np.zeros(T, dtype=np.float32)
    terminals[-1] = 1.0
    timeouts = _make_timeouts(T)

    obs_mean = observations.mean(axis=0).astype(np.float32)
    obs_std = observations.std(axis=0).astype(np.float32) + 1e-6
    act_mean = actions.mean(axis=0).astype(np.float32)
    act_std = actions.std(axis=0).astype(np.float32) + 1e-6

    light_ranges = np.array(
        [[0.0, LIGHTING_INSTALLED_W[zone]] for zone in ZONES_MID], dtype=np.float32)

    # ---- episode return (undiscounted) for quick scoring ----
    ep_return_physical_raw = float(reward_physical_raw.sum())
    ep_return_balanced = float(reward_balanced.sum())
    # Legacy training-compatible alias.
    ep_return_raw = ep_return_balanced
    ep_return_tx = float(rewards.sum())

    print("\n=== Agent evaluation dataset shapes ===")
    print(f"  observations:      {observations.shape}  (29-dim, meter-free)")
    print(f"  actions  (norm):   {actions.shape}      (agent executed, SDAR space)")
    print(f"  actions_raw:       {actions_raw.shape}  (physical units)")
    print(f"  prev_actions:      {prev_actions.shape}  (logged and verified)")
    print(f"  selection_logits:  {selection_logits.shape}  (fixed-rate active logits)")
    print(
        "  selection_probs:   "
        f"{selection_probabilities.shape}  (fixed target rates)"
    )
    print(f"  learned_probs:     {learned_selection_probabilities.shape}")
    mask_description = (
        "constant Bernoulli"
        if selector_mode == "constant"
        else "deterministic fractional-period"
    )
    print(f"  selection_masks:   {selection_masks.shape}  ({mask_description})")
    print(f"  executed_masks:    {executed_selection_masks.shape}")
    print(f"  actual_changes:    {actual_change_masks.shape}")
    print(f"  action_mixes:      {action_mixes.shape}  (reconstructed)")
    print(f"  rewards:           {rewards.shape}  (transform={REWARD_TRANSFORM}, normalize={REWARD_NORMALIZE})")
    print(f"  rewards_raw:       {rewards_raw.shape}  (balanced pre-transform)")
    print(f"  physical raw:      {reward_physical_raw.shape}  (sum of physical W_* terms)")
    print(f"  next_observations: {next_observations.shape}")
    print(f"  terminals:         {terminals.shape}  true terminals={int(terminals.sum())}")
    print(
        "  episode return:    "
        f"physical={ep_return_physical_raw:+.4f}  "
        f"balanced={ep_return_balanced:+.4f}  "
        f"transformed={ep_return_tx:+.4f}"
    )

    _print_normalization(
        comp_names,
        comp_scales,
        importance_vec,
        component_clips,
        scales_source,
    )
    _print_reward_diagnostics(
        reward_physical_raw,
        reward_balanced,
        rewards_tx,
        components_final,
    )
    _print_thermal_violation_rate()
    _print_switching_diagnostics(selection_masks)
    selector_diagnostics = _print_selector_probability_diagnostics(
        selection_probabilities,
        learned_selection_probabilities,
        selection_masks,
        executed_selection_masks,
        actual_change_masks,
        target_update_rates,
        force_first_update,
        selector_mode,
    )
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
            prev_actions=prev_actions,
            selection_logits=selection_logits,
            selection_probabilities=selection_probabilities,
            learned_selection_logits=learned_selection_logits,
            learned_selection_probabilities=learned_selection_probabilities,
            selection_masks=selection_masks,
            executed_selection_masks=executed_selection_masks,
            actual_change_masks=actual_change_masks,
            target_update_rates=target_update_rates,
            proposal_actions=proposal_actions,
            logged_prev_actions=logged_prev_actions,
            policy_executed_actions=policy_executed_actions,
            override_dims=override_dims,
            deadband_applied=deadband_applied,
            lighting_safety_active=lighting_safety_active,
            lighting_safety_override=lighting_safety_override,
            selector_modes=selector_modes,
            selector_stochastic=selector_stochastic,
            force_first_update=force_first_update,
            action_mixes=action_mixes,
            rewards=rewards,
            rewards_raw=rewards_raw,
            reward_balanced=reward_balanced,
            reward_physical_raw=reward_physical_raw,
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
            reward_transform=np.array(reward_reference["transform"]),
            reward_normalize=np.array(REWARD_NORMALIZE),
            reward_component_clip=np.array(
                -1.0 if REWARD_COMPONENT_CLIP is None else REWARD_COMPONENT_CLIP,
                dtype=np.float32),
            thermal_reward_component_clip=np.float32(
                component_clips[comp_names.index("thermal_comfort")]
            ),
            action_mask_value=np.float32(ACTION_MASK_VALUE),
            # physical lighting quantities (action-derived)
            lighting_dimming=lighting_dimming,
            lighting_power_from_action_w=lighting_power_from_action_w,
            lighting_energy_kwh=lighting_energy_kwh,
            light_ranges_per_zone=light_ranges,
            light_range_zone_order=np.array(list(ZONES_MID)),
            # evaluation provenance / quick scores
            policy=np.array(policy_tag),
            dataset_kind=np.array("sdar_ablation_agent_evaluation"),
            checkpoint_epoch=np.int64(checkpoint["epoch"]),
            checkpoint_path=np.array(checkpoint["path"]),
            checkpoint_sha256=np.array(checkpoint["sha256"]),
            eval_seed=np.int64(-1 if eval_seed is None else int(eval_seed)),
            eval_selector_mode=np.array(selector_mode),
            eval_proposal_deterministic=np.array(
                proposal_deterministic
            ),
            eval_selector_seed=np.int64(selector_seed),
            eval_periodic_phase_seed=np.int64(
                periodic_phase_seed if selector_mode == "periodic" else -1
            ),
            eval_periodic_schedule_verified=np.array(
                periodic_schedule_verified
            ),
            eval_fixed_update_rate_names=np.array(
                ["glazing", "lighting", "heating", "cooling"]
            ),
            eval_fixed_update_rates=np.array(
                [
                    fixed_update_rates["glazing"],
                    fixed_update_rates["lighting"],
                    fixed_update_rates["heating"],
                    fixed_update_rates["cooling"],
                ],
                dtype=np.float32,
            ),
            eval_fixed_update_rate_vector=expected_rate_vector,
            eval_safeguard_enabled=np.array(bool(safeguard_enabled)),
            eval_weather_file=np.array("" if weather_file is None else str(weather_file)),
            reward_reference_dataset=np.array(reward_reference["path"]),
            reward_reference_sha256=np.array(reward_reference["sha256"]),
            reward_reweighting_tag=np.array(reward_reference["reweighting_tag"]),
            episode_return_physical_raw=np.float32(ep_return_physical_raw),
            episode_return_balanced=np.float32(ep_return_balanced),
            episode_return_raw=np.float32(ep_return_raw),
            episode_return_transformed=np.float32(ep_return_tx),
            selector_diagnostic_groups=np.array(
                ["glazing", "lighting", "heating", "cooling", "overall"]
            ),
            selector_mean_probability_by_group=np.array(
                [
                    selector_diagnostics[name]["mean_probability"]
                    for name in (
                        "glazing",
                        "lighting",
                        "heating",
                        "cooling",
                        "overall",
                    )
                ],
                dtype=np.float32,
            ),
            selector_learned_mean_probability_by_group=np.array(
                [
                    selector_diagnostics[name]["learned_mean_probability"]
                    for name in (
                        "glazing",
                        "lighting",
                        "heating",
                        "cooling",
                        "overall",
                    )
                ],
                dtype=np.float32,
            ),
            selector_target_rate_by_group=np.array(
                [
                    selector_diagnostics[name]["target_rate"]
                    for name in (
                        "glazing",
                        "lighting",
                        "heating",
                        "cooling",
                        "overall",
                    )
                ],
                dtype=np.float32,
            ),
            selector_sampled_rate_by_group=np.array(
                [
                    selector_diagnostics[name]["sampled_rate"]
                    for name in (
                        "glazing",
                        "lighting",
                        "heating",
                        "cooling",
                        "overall",
                    )
                ],
                dtype=np.float32,
            ),
            selector_executed_update_rate_by_group=np.array(
                [
                    selector_diagnostics[name]["executed_update_rate"]
                    for name in (
                        "glazing",
                        "lighting",
                        "heating",
                        "cooling",
                        "overall",
                    )
                ],
                dtype=np.float32,
            ),
            selector_actual_change_rate_by_group=np.array(
                [
                    selector_diagnostics[name]["actual_change_rate"]
                    for name in (
                        "glazing",
                        "lighting",
                        "heating",
                        "cooling",
                        "overall",
                    )
                ],
                dtype=np.float32,
            ),
            selector_sample_minus_probability_by_group=np.array(
                [
                    selector_diagnostics[name]["difference"]
                    for name in (
                        "glazing",
                        "lighting",
                        "heating",
                        "cooling",
                        "overall",
                    )
                ],
                dtype=np.float32,
            ),
            selector_threshold_agreement=np.float32(
                np.nan
                if selector_diagnostics["threshold_agreement"] is None
                else selector_diagnostics["threshold_agreement"]
            ),
            action_layout=np.array(
                {"glazing": [0, 4], "lighting": [4, 9], "heating": [9, 14], "cooling": [14, 19]},
                dtype=object),
        )

        if SAVE_REWARD_COMPONENTS:
            save_dict["reward_components"] = components_final
            save_dict["reward_components_unbounded"] = components_unbounded
            save_dict["reward_components_raw"] = components_raw
            save_dict["reward_component_names"] = np.array(comp_names)
            save_dict["reward_component_scales"] = comp_scales.astype(np.float32)
            save_dict["reward_component_importance"] = importance_vec.astype(np.float32)
            save_dict["reward_component_clips"] = component_clips.astype(np.float32)

        np.savez_compressed(npz_path, **save_dict)
        print(f"\nSaved NPZ: {npz_path}")

        stats_path = f"{prefix}_stats.json"
        with open(stats_path, "w") as f:
            json.dump(
                {
                    "policy": policy_tag,
                    "dataset_kind": "sdar_ablation_agent_evaluation",
                    "checkpoint": checkpoint,
                    "evaluation": {
                        "seed": None if eval_seed is None else int(eval_seed),
                        "selector_mode": selector_mode,
                        "proposal_deterministic": proposal_deterministic,
                        "selector_seed": selector_seed,
                        "periodic_phase_seed": (
                            periodic_phase_seed
                            if selector_mode == "periodic"
                            else None
                        ),
                        "periodic_schedule_verified": (
                            periodic_schedule_verified
                            if selector_mode == "periodic"
                            else None
                        ),
                        "fixed_update_rates": fixed_update_rates,
                        "safeguard_enabled": bool(safeguard_enabled),
                        "weather_file": None if weather_file is None else str(weather_file),
                    },
                    "reward_reference": {
                        "dataset": reward_reference["path"],
                        "sha256": reward_reference["sha256"],
                        "reweighting_tag": reward_reference["reweighting_tag"],
                    },
                    "reward_transform": reward_reference["transform"],
                    "reward_normalize": bool(REWARD_NORMALIZE),
                    "reward_component_clip": REWARD_COMPONENT_CLIP,
                    "thermal_reward_component_clip": float(
                        component_clips[comp_names.index("thermal_comfort")]
                    ),
                    "reward_scales_source": scales_source,
                    "n_transitions": int(T),
                    "obs_dim": int(observations.shape[1]),
                    "act_dim": int(actions.shape[1]),
                    "episode_return_physical_raw": ep_return_physical_raw,
                    "episode_return_balanced": ep_return_balanced,
                    "episode_return_raw": ep_return_raw,
                    "episode_return_transformed": ep_return_tx,
                    "reward_physical_raw_mean": float(reward_physical_raw.mean()),
                    "reward_physical_raw_std": float(reward_physical_raw.std()),
                    "reward_balanced_mean": float(reward_balanced.mean()),
                    "reward_balanced_std": float(reward_balanced.std()),
                    "reward_raw_mean": float(rewards_raw.mean()),
                    "reward_raw_std": float(rewards_raw.std()),
                    "reward_raw_min": float(rewards_raw.min()),
                    "reward_raw_max": float(rewards_raw.max()),
                    "reward_tx_mean": float(rewards_tx.mean()),
                    "reward_tx_std": float(rewards_tx.std()),
                    "reward_component_names": comp_names,
                    "reward_component_scales": comp_scales.tolist(),
                    "reward_component_importance": importance_vec.tolist(),
                    "reward_component_clips": component_clips.tolist(),
                    "comfort_band": {
                        "low_c": COMFORT_TEMP_LOW,
                        "high_c": COMFORT_TEMP_HIGH,
                        "occupied_only": COMFORT_OCCUPIED_ONLY,
                    },
                    "mean_selection_mask_overall": float(selection_masks.mean()),
                    "mean_selection_mask_glazing": float(selection_masks[:, GLAZE_SLICE].mean()),
                    "mean_selection_mask_lighting": float(selection_masks[:, LIGHT_SLICE].mean()),
                    "mean_selection_mask_heating": float(selection_masks[:, HEAT_SLICE].mean()),
                    "mean_selection_mask_cooling": float(selection_masks[:, COOL_SLICE].mean()),
                    "mean_executed_selection_mask_overall": float(
                        executed_selection_masks.mean()
                    ),
                    "mean_actual_change_mask_overall": float(
                        actual_change_masks.mean()
                    ),
                    "mean_actual_change_mask_glazing": float(
                        actual_change_masks[:, GLAZE_SLICE].mean()
                    ),
                    "mean_actual_change_mask_lighting": float(
                        actual_change_masks[:, LIGHT_SLICE].mean()
                    ),
                    "mean_actual_change_mask_heating": float(
                        actual_change_masks[:, HEAT_SLICE].mean()
                    ),
                    "mean_actual_change_mask_cooling": float(
                        actual_change_masks[:, COOL_SLICE].mean()
                    ),
                    "mean_learned_selection_probability_overall": float(
                        learned_selection_probabilities.mean()
                    ),
                    "override_dimension_rate": float(override_dims.mean()),
                    "deadband_step_rate": float(
                        np.any(deadband_applied > 0.5, axis=1).mean()
                    ),
                    "lighting_safety_active_step_rate": float(
                        lighting_safety_active.mean()
                    ),
                    "lighting_safety_override_step_rate": float(
                        lighting_safety_override.mean()
                    ),
                    "selector_probability_diagnostics": selector_diagnostics,
                    "hvac_demand_rate_mean_w": float(hvac_demand_rate.mean()),
                    "hvac_demand_rate_p95_w": float(np.quantile(hvac_demand_rate, 0.95)),
                    "hvac_demand_rate_max_w": float(hvac_demand_rate.max()),
                    "light_installed_w": {z: LIGHTING_INSTALLED_W[z] for z in ZONES_MID},
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
        mask_cols = [f"selected_{c}" for c in act_cols]
        executed_mask_cols = [f"executed_update_{c}" for c in act_cols]
        change_mask_cols = [f"actual_change_{c}" for c in act_cols]
        override_cols = [f"override_{c}" for c in act_cols]
        logit_cols = [f"active_selector_logit_{c}" for c in act_cols]
        probability_cols = [
            f"active_selector_probability_{c}"
            for c in act_cols
        ]
        learned_logit_cols = [f"learned_selector_logit_{c}" for c in act_cols]
        learned_probability_cols = [
            f"learned_selector_probability_{c}" for c in act_cols
        ]
        target_rate_cols = [f"target_update_rate_{c}" for c in act_cols]
        proposal_cols = [f"proposal_norm_{c}" for c in act_cols]
        previous_cols = [f"previous_norm_{c}" for c in act_cols]
        policy_action_cols = [f"policy_executed_norm_{c}" for c in act_cols]
        executed_action_cols = [f"executed_norm_{c}" for c in act_cols]
        deadband_cols = [f"deadband_applied_{zone}" for zone in ZONES_MID]
        next_obs_cols = [f"next_{c}" for c in obs_cols]

        header = (
            obs_cols
            + act_cols
            + mask_cols
            + executed_mask_cols
            + change_mask_cols
            + override_cols
            + logit_cols
            + probability_cols
            + learned_logit_cols
            + learned_probability_cols
            + target_rate_cols
            + proposal_cols
            + previous_cols
            + policy_action_cols
            + executed_action_cols
            + deadband_cols
            + [
                "lighting_safety_active",
                "lighting_safety_override",
                "selector_stochastic",
                "force_first_update",
            ]
            + ["reward", "reward_raw", "reward_physical_raw"]
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
                    + selection_masks[t].tolist()
                    + executed_selection_masks[t].tolist()
                    + actual_change_masks[t].tolist()
                    + override_dims[t].tolist()
                    + selection_logits[t].tolist()
                    + selection_probabilities[t].tolist()
                    + learned_selection_logits[t].tolist()
                    + learned_selection_probabilities[t].tolist()
                    + target_update_rates[t].tolist()
                    + proposal_actions[t].tolist()
                    + logged_prev_actions[t].tolist()
                    + policy_executed_actions[t].tolist()
                    + actions[t].tolist()
                    + deadband_applied[t].tolist()
                    + [
                        int(lighting_safety_active[t]),
                        int(lighting_safety_override[t]),
                        int(selector_stochastic[t]),
                        int(force_first_update[t]),
                    ]
                    + [
                        float(rewards[t]),
                        float(rewards_raw[t]),
                        float(reward_physical_raw[t]),
                    ]
                    + (components_final[t].tolist() if SAVE_REWARD_COMPONENTS else [])
                    + next_observations[t].tolist()
                    + [float(terminals[t]), float(timeouts[t])]
                )
                writer.writerow(row)
        print(f"Saved CSV: {csv_path}  ({T} rows, {len(header)} cols)")

    return {
        "observations": observations,
        "actions": actions,
        "actions_raw": actions_raw,
        "prev_actions": prev_actions,
        "selection_logits": selection_logits,
        "selection_probabilities": selection_probabilities,
        "learned_selection_logits": learned_selection_logits,
        "learned_selection_probabilities": learned_selection_probabilities,
        "selection_masks": selection_masks,
        "executed_selection_masks": executed_selection_masks,
        "actual_change_masks": actual_change_masks,
        "target_update_rates": target_update_rates,
        "proposal_actions": proposal_actions,
        "logged_prev_actions": logged_prev_actions,
        "policy_executed_actions": policy_executed_actions,
        "override_dims": override_dims,
        "deadband_applied": deadband_applied,
        "lighting_safety_active": lighting_safety_active,
        "lighting_safety_override": lighting_safety_override,
        "selector_modes": selector_modes,
        "selector_stochastic": selector_stochastic,
        "force_first_update": force_first_update,
        "action_mixes": action_mixes,
        "rewards": rewards,
        "rewards_raw": rewards_raw,
        "reward_balanced": reward_balanced,
        "reward_physical_raw": reward_physical_raw,
        "reward_components": components_final,
        "reward_components_unbounded": components_unbounded,
        "reward_components_raw": components_raw,
        "reward_component_names": comp_names,
        "reward_component_scales": comp_scales,
        "reward_component_importance": importance_vec,
        "reward_component_clips": component_clips,
        "next_observations": next_observations,
        "terminals": terminals,
        "timeouts": timeouts,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "act_mean": act_mean,
        "act_std": act_std,
        "light_ranges_per_zone": light_ranges,
        "lighting_dimming": lighting_dimming,
        "lighting_power_from_action_w": lighting_power_from_action_w,
        "lighting_energy_kwh": lighting_energy_kwh,
        "checkpoint": checkpoint,
        "evaluation": {
            "seed": None if eval_seed is None else int(eval_seed),
            "selector_mode": selector_mode,
            "proposal_deterministic": proposal_deterministic,
            "selector_seed": selector_seed,
            "periodic_phase_seed": (
                periodic_phase_seed if selector_mode == "periodic" else None
            ),
            "periodic_schedule_verified": (
                periodic_schedule_verified
                if selector_mode == "periodic"
                else None
            ),
            "fixed_update_rates": fixed_update_rates,
            "safeguard_enabled": bool(safeguard_enabled),
            "weather_file": (
                None if weather_file is None else str(weather_file)
            ),
        },
        "selector_probability_diagnostics": selector_diagnostics,
        "reward_reference": reward_reference,
        "episode_return_physical_raw": ep_return_physical_raw,
        "episode_return_balanced": ep_return_balanced,
        "episode_return_raw": ep_return_raw,
        "episode_return_transformed": ep_return_tx,
    }


if __name__ == "__main__":
    dataset = build_agent_dataset(
        save_npz=True,
        save_csv=True,
        reference_dataset_path=REFERENCE_REWARD_DATASET,
        checkpoint_path=EVAL_CHECKPOINT_PATH,
        checkpoint_epoch=EVAL_CHECKPOINT_EPOCH,
        eval_seed=EVAL_SEED,
        selector_mode=EVAL_SELECTOR_MODE,
        proposal_deterministic=EVAL_PROPOSAL_DETERMINISTIC,
        selector_seed=EVAL_SELECTOR_SEED,
        fixed_update_rates=EVAL_FIXED_UPDATE_RATES,
        periodic_phase_seed=EVAL_PERIODIC_PHASE_SEED,
        safeguard_enabled=EVAL_SAFEGUARD_ENABLED,
        weather_file=EVAL_WEATHER_FILE,
    )
