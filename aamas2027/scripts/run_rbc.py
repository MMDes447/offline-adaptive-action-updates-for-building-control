#!/usr/bin/env python3
"""Annual run of the behaviour controller (clean rule-based controller, RBC)
with the project's own notebook code, verbatim, under any weather file.

    python aamas2027/scripts/run_rbc.py --run-name rbc_clean_sf
    python aamas2027/scripts/run_rbc.py --weather usa_ca_fresno --run-name rbc_clean_fresno

Which code runs (frozen byte-for-byte in faithful/cells/, checked against
faithful/CELLS.json before every run):

  new_dataset_gen1.ipynb
    cells 0-4   frads medium office, SageGlass SR2 glazing systems, lighting,
                EnergyPlusSetup (the same cells run_eval.py uses for the agent)
    cell 8      v12 smooth expert + callback-side exploration: the controller that
                generated all 14 data episodes. With NOISE_PROFILE = "clean" the
                exploration and the forced repetition are switched off
                (EXPLORATION_ENABLED = False), so the controller is deterministic.
    cell 16/17  eps.set_callback(... callback_func) and eps.run(annual=True)
    cell 33     expert-episode builder (default prefix offline_smooth_clean_episode_000,
                i.e. the code that wrote the reference RBC files)

Deviations from running the notebook by hand:
  1. literal lines of cell 8 are substituted, exactly as one would edit them by hand
     for the clean episode: EPISODE_ID = 13 -> 0, NOISE_PROFILE = "high" -> "clean",
     and (default --band data) the thermostat band 22-25/24 C -> 21-24/22 C that the
     data episodes 0-12 were collected with (see BAND_DATA); each must occur once;
  2. the weather argument of cell 4 is replaced for non-SF weather (run_eval.py);
  3. EnergyPlus writes to runs/<name>/eplus/, the builder writes runs/<name>/rollout.*;
  4. with --check-reference (San Francisco only) the annual KPIs are compared with
     the original episode; a mismatch marks the run failed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE))
import run_eval as R  # noqa: E402  (shared helpers: frozen cells, weather swap, versions)

EXPERT_CELL = "gen1_08_expert_v12.py"
BUILDER_CELL = "gen1_33_expert_builder.py"
SUBST = [("EPISODE_ID = 13 ", "EPISODE_ID = 0  "),
         ('NOISE_PROFILE = "high" ', 'NOISE_PROFILE = "clean"')]
# Thermostat band. The notebook cell now has COMFORT_BAND 22-25 C / override 24 C, but
# episodes 0-12 of the dataset (incl. the clean reference episode) were collected with
# 21-24 C / 22 C: replaying their logged zone temperatures through the cell's thermal rule
# reproduces 100% of the clean episode's heating setpoints with 21-24 and 49% with 22-25;
# the noisy episodes' idle shares inside 21-22 C match their profiles only with 21-24.
# Only episode 13 (high noise, the last one collected) used 22-25 C.
BAND_DATA = [("COMFORT_BAND_LOW    = 22.0", "COMFORT_BAND_LOW    = 21.0"),
             ("COMFORT_BAND_HIGH   = 25.0", "COMFORT_BAND_HIGH   = 24.0"),
             ("COMFORT_OVERRIDE_SP = 24.0", "COMFORT_OVERRIDE_SP = 22.0")]
REFERENCE_RBC = "offline_smooth_clean_episode_000.csv"
TOL = {"combined_kwh_day": ("rel", 0.01), "tv_pct": ("abs", 0.3), "vis_in_pct": ("abs", 0.5),
       "upd_pct": ("abs", 0.3)}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weather", default=R.DEFAULT_WEATHER, help="pyenergyplus weather key or .epw path")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--runs-dir", default=str(R.AAMAS_DIR / "runs"))
    ap.add_argument("--check-reference", action="store_true",
                    help=f"compare annual KPIs with {REFERENCE_RBC} (San Francisco weather only)")
    ap.add_argument("--band", choices=("data", "notebook"), default="data",
                    help="data: 21-24 C band of the data-collection controller (default); "
                         "notebook: the 22-25 C band currently in the notebook cell (a retuned RBC)")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args(argv)


def kpis_of(csv_path: Path) -> dict:
    sys.path.insert(0, str(R.PKG))
    import kpis as K
    return K.annual(K.load(csv_path))


def main(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    os.chdir(R.PROJECT_ROOT)
    sys.path.insert(0, str(R.PROJECT_ROOT))
    run_dir = Path(a.runs_dir) / a.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not a.overwrite:
        sys.exit(f"ERROR: {run_dir} already exists; choose another --run-name or pass --overwrite")
    (run_dir / "eplus").mkdir(parents=True, exist_ok=True)
    if a.check_reference and (a.weather != R.DEFAULT_WEATHER or a.band != "data"):
        sys.exit("ERROR: --check-reference needs San Francisco weather and --band data")
    subst = SUBST + (BAND_DATA if a.band == "data" else [])

    manifest = {
        "status": "running", "kind": "rbc_clean", "run_name": a.run_name,
        "command": " ".join([sys.executable] + sys.argv),
        "started": _dt.datetime.now().isoformat(timespec="seconds"),
        "project_root": str(R.PROJECT_ROOT), "args": vars(a),
        "substitutions": subst, "band": a.band,
        "evaluation_code": {"run_rbc.py": R.sha256(__file__), **R.code_hashes()},
        "notebooks": json.loads((R.FAITHFUL / "CELLS.json").read_text())["notebooks"],
        "versions": R.versions(),
        # make_* scripts only read runs whose provenance matched; for the RBC there is
        # no checkpoint, so "match" records that the frozen controller code was used.
        "provenance": {"match": True, "kind": "rbc_frozen_cells", "expert_cell": EXPERT_CELL},
    }
    mpath = run_dir / "manifest.json"

    def write_manifest():
        manifest["updated"] = _dt.datetime.now().isoformat(timespec="seconds")
        tmp = mpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, default=str))
        tmp.replace(mpath)

    try:
        write_manifest()
        ns = {"__name__": "sdar_eval_namespace"}                       # builder's __main__ guard stays off
        manifest["simulation"] = R.build_simulation(ns, a.weather, 0)   # cells 0-4 (+ weather swap)
        write_manifest()

        src = R.load_cell(EXPERT_CELL)                                  # cell 8
        for old, new in subst:
            if src.count(old) != 1:
                raise RuntimeError(f"cannot find exactly one {old!r} in {EXPERT_CELL}")
            src = src.replace(old, new)
        R.exec_source(src, str(R.CELLS / EXPERT_CELL), ns)
        if ns["NOISE_PROFILE"] != "clean" or ns["EXPLORATION_ENABLED"] or ns["EPISODE_ID"] != 0:
            raise RuntimeError("expert cell is not in the clean configuration")
        manifest["thermostat"] = {k: ns[k] for k in ("COMFORT_BAND_LOW", "COMFORT_BAND_HIGH",
                                                      "COMFORT_OVERRIDE_SP", "THERMAL_TARGET_SP")}
        eps = ns["eps"]
        eps.set_callback("callback_begin_system_timestep_before_predictor", ns["callback_func"])  # cell 16
        t_sim = time.time()
        eps.run(output_directory=str(run_dir / "eplus"), annual=True)                             # cell 17
        manifest["timing"] = {"simulation_s": round(time.time() - t_sim, 1)}
        first, last = R.energyplus_version(run_dir / "eplus")
        manifest["energyplus"] = {"version_line": first, "completion_line": last}
        write_manifest()

        R.exec_source(R.load_cell(BUILDER_CELL), str(R.CELLS / BUILDER_CELL), ns)  # cell 33
        ns["build_offline_dataset"](save_npz=True, save_csv=True, prefix=str(run_dir / "rollout"))

        csv = run_dir / "rollout.csv"
        manifest["kpis"] = kpis_of(csv)
        if a.check_reference:
            ref = kpis_of(R.PROJECT_ROOT / REFERENCE_RBC)
            rep = {"reference": REFERENCE_RBC, "ok": True, "diffs": {}}
            for k, (kind, tol) in TOL.items():
                new, old = manifest["kpis"].get(k), ref.get(k)
                d = (new - old) / abs(old) if kind == "rel" else new - old
                rep["diffs"][k] = {"new": new, "reference": old, "diff": d, "tol": tol, "kind": kind}
                rep["ok"] &= abs(d) <= tol
            manifest["reproduction"] = rep
            write_manifest()
            if not rep["ok"]:
                raise RuntimeError("RBC does not reproduce the reference clean episode: "
                                   + json.dumps({k: round(v["diff"], 4) for k, v in rep["diffs"].items()}))
        manifest["outputs"] = {p.name: {"sha256": R.sha256(p), "bytes": p.stat().st_size}
                               for p in sorted(run_dir.glob("rollout*")) if p.is_file()}
        manifest["status"] = "ok"
    except BaseException as e:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(e).__name__}: {e}"
        manifest["traceback"] = traceback.format_exc()
        write_manifest()
        print(manifest["traceback"], file=sys.stderr)
        sys.exit(1)
    finally:
        manifest["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
        manifest.setdefault("timing", {})["total_s"] = round(time.time() - t0, 1)
        write_manifest()
    k = manifest["kpis"]
    print(f"\nOK  {a.run_name}: {k['combined_kwh_day']:.1f} kWh/day, {k['tv_pct']:.2f}% violations, "
          f"{k['vis_in_pct']:.1f}% visual in band; outputs in {run_dir}")


if __name__ == "__main__":
    main()
