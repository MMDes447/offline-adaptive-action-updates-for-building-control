"""Same KPIs as kpis.annual(), computed from rollout.npz instead of rollout.csv
(the NPZ is ~10x smaller). Checked to agree with kpis.annual on both builder formats."""
import numpy as np
import pandas as pd
import kpis as K


def frame(path):
    d = np.load(path, allow_pickle=True)
    o = d["observations"]
    df = pd.DataFrame({f"temp_{z}": o[:, i] for i, z in enumerate(K.ZONES)})
    lp = d["lighting_power_from_action_w"]
    for i, z in enumerate(K.ZONES):
        df[f"light_power_{z}"] = lp[:, i]
    for i, z in enumerate(K.ZONES[:4]):
        df[f"wpi_{z}"] = o[:, 18 + i]
        df[f"ext_irr_{z}"] = o[:, 14 + i]
        df[f"glazing_{z}"] = np.rint((d["actions"][:, i] + 1.0) * 1.5)
    df["is_occupied"] = o[:, 27]
    df["hvac_electricity_demand_rate"] = d["hvac_demand_rate"]
    m = d["selection_masks"]
    groups = ["glazing"] * 4 + ["light_power"] * 5 + ["heat_sp"] * 5 + ["cool_sp"] * 5
    zones = K.ZONES[:4] + K.ZONES * 3
    for j in range(19):
        df[f"mask_{groups[j]}_{zones[j]}"] = m[:, j]
    return df


def annual(path):
    return K.annual(frame(path))


def daily(path):
    return K.daily(frame(path))
