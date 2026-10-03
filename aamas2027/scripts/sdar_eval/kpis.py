"""Physical KPIs for annual rollout CSVs.

The definitions are identical to res_plot.py and reproduce its numbers:

- occupied thermal comfort band: 21-24 C;
- perimeter illuminance = WPI + 0.4 lux/W * lighting power, in band at 450-550 lux;
- daylight glare = WPI > 1000 lux;
- core illuminance = 0.4 * core lighting power;
- HVAC energy = Facility Total HVAC Electricity Demand Rate sampled every 10 min.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ZONES = ["Perimeter_mid_ZN_1", "Perimeter_mid_ZN_2", "Perimeter_mid_ZN_3",
         "Perimeter_mid_ZN_4", "Core_mid"]
SPH = 6
T_LO, T_HI = 21.0, 24.0
LUX_PER_W, LUX_LO, LUX_HI, GLARE = 0.4, 450.0, 550.0, 1000.0
GROUPS = {"glazing": "glazing", "lighting": "light_power", "heating": "heat_sp", "cooling": "cool_sp"}


def _cols(df, prefix):
    return [f"{prefix}{z}" for z in ZONES if f"{prefix}{z}" in df]


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "light_power_Perimeter_mid_ZN_1" not in df:
        df = df.rename(columns=lambda c: c.replace("lighting_power_action_", "light_power_"))
    return df


def mask_frame(df):
    m = [c for c in df.columns if c.startswith("mask_")]
    if len(m) != 19:
        m = [c for c in df.columns if c.startswith("selected_")]
    return df[m] if len(m) == 19 else None


def step_arrays(df):
    light = df[_cols(df, "light_power_")].to_numpy(float)
    temp = df[_cols(df, "temp_")].to_numpy(float)
    wpi = df[[f"wpi_{z}" for z in ZONES[:4]]].to_numpy(float)
    occ = df["is_occupied"].to_numpy(float) > 0.5
    hvac = df["hvac_electricity_demand_rate"].to_numpy(float)
    return light, temp, wpi, occ, hvac


def annual(df) -> dict:
    light, temp, wpi, occ, hvac = step_arrays(df)
    days = len(df) / SPH / 24.0
    out = {}
    out["lighting_kwh_day"] = light.sum() / SPH / 1000 / days
    out["hvac_kwh_day"] = hvac.sum() / SPH / 1000 / days
    out["combined_kwh_day"] = out["lighting_kwh_day"] + out["hvac_kwh_day"]
    t = temp[occ]
    out["tv_pct"] = 100 * ((t < T_LO) | (t > T_HI)).mean()
    out["cold_pct"] = 100 * (t < T_LO).mean()
    out["hot_pct"] = 100 * (t > T_HI).mean()
    out["dzh_day"] = (np.maximum(T_LO - t, 0) + np.maximum(t - T_HI, 0)).sum() / SPH / days
    tot = (wpi + LUX_PER_W * light[:, :4])[occ]
    out["vis_in_pct"] = 100 * ((tot >= LUX_LO) & (tot <= LUX_HI)).mean()
    out["vis_under_pct"] = 100 * (tot < LUX_LO).mean()
    out["vis_over_pct"] = 100 * (tot > LUX_HI).mean()
    out["glare_pct"] = 100 * (wpi[occ] > GLARE).mean()
    core = LUX_PER_W * light[occ, 4]
    out["core_in_pct"] = 100 * ((core >= LUX_LO) & (core <= LUX_HI)).mean()
    out["unocc_light_kwh_day"] = light[~occ].sum() / SPH / 1000 / days
    gcols = [f"glazing_{z}" for z in ZONES[:4]]
    if all(c in df for c in gcols):
        g = df[gcols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        irr = df[[f"ext_irr_{z}" for z in ZONES[:4]]].to_numpy(float)
        sunny = (irr > 300) & occ[:, None]
        out["glz_clear_pct"] = 100 * (g[occ] == 0).mean()
        out["glz_dark_pct"] = 100 * (g[occ] == 3).mean()
        out["glz_clear_sunny_pct"] = 100 * (g[sunny] == 0).mean()
    m = mask_frame(df)
    if m is not None:
        mv = m.to_numpy(float)
        out["upd_pct"] = 100 * mv.mean()
        out["persistence_steps"] = 1.0 / max(mv.mean(), 1e-9)
        for g, key in GROUPS.items():
            out[f"upd_{g}_pct"] = 100 * m[[c for c in m.columns if f"_{key}_" in c]].to_numpy(float).mean()
    return out


def daily(df) -> pd.DataFrame:
    """Per-day series for paired, day-by-day comparisons of two runs.

    tv_pct is each day's share of the annual occupied zone-steps that violate the
    band, scaled so that its mean over all days equals annual()["tv_pct"]
    (unoccupied days contribute 0). Occupancy follows the same schedule in every
    run, so the mean of a paired daily difference equals the difference of the
    annual percentages reported in the tables. tv_pct_day is the plain within-day share.
    """
    light, temp, wpi, occ, hvac = step_arrays(df)
    day = np.arange(len(df)) // (SPH * 24)
    n_days = len(np.unique(day))
    n_occ = temp[occ].size
    rows = []
    for d in np.unique(day):
        s = day == d
        o = occ[s]
        t = temp[s][o]
        nviol = ((t < T_LO) | (t > T_HI)).sum()
        rows.append({
            "day": int(d),
            "combined_kwh": (light[s].sum() + hvac[s].sum()) / SPH / 1000,
            "tv_pct": 100 * nviol * n_days / n_occ if n_occ else np.nan,
            "tv_pct_day": 100 * nviol / t.size if o.any() else np.nan,
            "dzh": (np.maximum(T_LO - t, 0) + np.maximum(t - T_HI, 0)).sum() / SPH,
            "occ_steps": int(o.sum()),
        })
    return pd.DataFrame(rows).set_index("day")


def block_bootstrap_mean(x, block=30, n_boot=4000, seed=0):
    """Moving-block bootstrap 95% CI of the mean of a (daily) series."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, n - block + 1, nb)
        means[b] = x[(st[:, None] + np.arange(block)).ravel()[:n]].mean()
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
