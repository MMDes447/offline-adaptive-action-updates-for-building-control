#!/usr/bin/env python3
"""Annual EnergyPlus/Radiance evaluation of one SDAR-IQL checkpoint, using the
project's own notebook code verbatim.

    python aamas2027/scripts/run_eval.py \
        --checkpoint checkpoints/<run>/<file>_epoch100.pt \
        --mode sample --selector-seed 20260728 --run-name b3e100_sample_s20260728

Which code runs (frozen byte-for-byte in faithful/cells/, checked against
faithful/CELLS.json before every run):

  mode sample | threshold   -> new_dataset_gen1.ipynb
        cells 0-4 (frads model, glazing, lighting, EnergyPlusSetup),
        cell 14 (evaluation callback, sdar_iql_train_3_updated.load_and_act),
        cell 46 (rollout builder).
        This is the code that produced every existing beta3 rollout.
  mode clock | always_prev (batch 7) -> as constant/periodic/always, but the agent is
        loaded through sdar_eval/policy_ext.py (exact substitutions on
        sdar_iql_train_4_updated.load_and_act): clock = hour-of-day Bernoulli selector
        from --hourly-rates; always_prev = update everything, proposal sees the
        previous action (full-action IQL baseline trained with --algo iql_flat_prev).
  mode constant | periodic | always -> Ablation_sdar_.ipynb
        cells 0-4 (identical to the above), cell 5 (ablation callback,
        sdar_iql_train_4_updated.load_and_act) and cell 9 (fixed-rate builder),
        with the small patches listed in sdar_eval/PATCHES.md.
        This is the code that produced the existing selector-ablation rollouts.

Deviations from running the notebook by hand (all in faithful/README.md):
  1. parameters (checkpoint, selector mode/seed, safeguard) are set on the
     executed namespace instead of by editing the cell text;
  2. EnergyPlus writes to runs/<name>/eplus/ instead of the project root;
  3. the builder is called with prefix runs/<name>/rollout instead of its
     default file name, so no existing file in the project root is touched;
  4. after the run the weight-level provenance check confirms that the logged
     selector logits came from --checkpoint.
--smoke-days N (tests only) stops EnergyPlus after the first N days.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent   # no symlink resolution: project root = two levels up
PKG = HERE / "sdar_eval"
FAITHFUL = HERE / "faithful"
CELLS = FAITHFUL / "cells"
AAMAS_DIR = HERE.parent
PROJECT_ROOT = AAMAS_DIR.parent

DEFAULT_WEATHER = "usa_ca_san_francisco"
DEFAULT_REFERENCE = "pooled_sdar_experts_thermal3_tclip50.npz"
DEFAULT_REFERENCE_SHA = "5c6c255e3b7791c4c31b76ea865a36a1933edb3b443a1406970056c7582cadfb"
MODES = ("sample", "threshold", "constant", "periodic", "always", "clock", "always_prev")
GEN1_MODES = ("sample", "threshold")
SETUP_CELLS = ("gen1_00_imports.py", "gen1_01_model.py", "gen1_02_glazing.py",
               "gen1_03_add_glazing_lighting.py", "gen1_04_energyplus_setup.py")
WEATHER_TOKEN = 'weather_files["usa_ca_san_francisco"]'


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ frozen cells
def load_cell(name: str) -> str:
    """Return a frozen notebook cell, refusing to run if it was edited."""
    meta = json.loads((FAITHFUL / "CELLS.json").read_text())["cells"][name]
    data = (CELLS / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != meta["sha256"]:
        raise RuntimeError(f"{name} no longer matches {meta['notebook']} cell {meta['cell']} "
                           "(re-run faithful/extract_cells.py if the notebook changed on purpose)")
    return data.decode("utf-8")


def exec_source(src: str, filename: str, ns: dict):
    # compile against the real file: frads' set_callback reads the callback's
    # source with inspect.getsource, which needs a file on disk
    exec(compile(src, filename, "exec"), ns)


def build_simulation(ns: dict, weather: str, smoke_days: int, log=print) -> dict:
    """Notebook cells 0-4, verbatim, in the shared namespace. Returns info."""
    for name in SETUP_CELLS[:4]:
        exec_source(load_cell(name), str(CELLS / name), ns)
    src = load_cell(SETUP_CELLS[4])
    epw_key = weather or DEFAULT_WEATHER
    if epw_key != DEFAULT_WEATHER:
        if src.count(WEATHER_TOKEN) != 1:
            raise RuntimeError("cannot locate the weather argument in cell 4")
        wf = ns["weather_files"]
        target = str(wf[epw_key]) if epw_key in wf else str(Path(epw_key).expanduser().resolve())
        if not Path(target).is_file():
            raise FileNotFoundError(f"weather {epw_key!r} not found")
        src = src.replace(WEATHER_TOKEN, repr(target))
        log(f"[run_eval] weather replaced: {target}")
    exec_source(src, str(CELLS / SETUP_CELLS[4]), ns)
    epw = str(ns["weather_files"][DEFAULT_WEATHER]) if epw_key == DEFAULT_WEATHER else target
    return {"weather_arg": epw_key, "weather_epw": epw, "weather_epw_sha256": sha256(epw),
            "setup_cells": list(SETUP_CELLS), "smoke_days": int(smoke_days)}


# ------------------------------------------------------------------ helpers
def parse_rates(text: str | None):
    if text is None:
        return None
    parts = [float(x) for x in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--rates needs glazing,lighting,heating,cooling")
    return dict(zip(("glazing", "lighting", "heating", "cooling"), parts))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="path relative to the project root, or absolute")
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--selector-seed", type=int, default=20260728)
    ap.add_argument("--phase-seed", type=int, default=None, help="periodic phase seed (default: selector seed)")
    ap.add_argument("--rates", type=parse_rates, default=None,
                    help="glazing,lighting,heating,cooling target update rates (constant/periodic)")
    ap.add_argument("--hourly-rates", default=None,
                    help="JSON file with a 24 x 19 table of update probabilities (mode clock)")
    ap.add_argument("--safeguard", choices=("on", "off"), default="on",
                    help="unoccupied lights-off safeguard (the heat<=cool deadband is always applied)")
    ap.add_argument("--reward-reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--reward-reference-sha", default=DEFAULT_REFERENCE_SHA,
                    help="expected sha256 of the reward reference; pass '' to skip")
    ap.add_argument("--expect-checkpoint-sha", default="", help="optional expected checkpoint sha256")
    ap.add_argument("--weather", default=DEFAULT_WEATHER, help="pyenergyplus weather key or .epw path")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--runs-dir", default=str(AAMAS_DIR / "runs"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--provenance-steps", type=int, default=2000)
    ap.add_argument("--smoke-days", type=int, default=0,
                    help="TEST ONLY: simulate Jan 1..N instead of the full year")
    a = ap.parse_args(argv)
    if a.mode in ("constant", "periodic") and a.rates is None:
        ap.error(f"--mode {a.mode} requires --rates")
    if a.mode == "clock" and a.hourly_rates is None:
        ap.error("--mode clock requires --hourly-rates")
    if a.phase_seed is None:
        a.phase_seed = a.selector_seed
    return a


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform(),
           "hostname": socket.gethostname(), "conda_env": os.environ.get("CONDA_DEFAULT_ENV")}
    for mod in ("numpy", "torch", "frads", "pyradiance", "pywincalc", "pandas"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:  # pragma: no cover
            out[mod] = f"unavailable: {e!r}"
    try:
        import torch
        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            out["cuda_device"] = torch.cuda.get_device_name(0)
            out["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    except Exception:
        pass
    try:
        out["git_commit"] = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                                           capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        out["git_commit"] = None
    return out


def energyplus_version(eplus_dir: Path):
    cands = [eplus_dir / "eplusout.err"] + sorted(eplus_dir.glob("*.err"))
    err = next((c for c in cands if c.is_file()), None)
    if err is None:
        return None, None
    lines = err.read_text(errors="replace").splitlines()
    return (lines[0].strip() if lines else None), (lines[-1].strip() if lines else None)


# ------------------------------------------------------------------ pipelines
def prepare_agent(ns: dict, a, ckpt: Path, rates):
    """Execute the callback cell and load the agent. Returns the code record."""
    if a.mode in GEN1_MODES:
        exec_source(load_cell("gen1_14_eval_callback.py"), str(CELLS / "gen1_14_eval_callback.py"), ns)
        # the cell's own module-level parameters; setup() reads them at call time
        ns["EVAL_CHECKPOINT"] = str(ckpt)
        ns["SELECTOR_MODE"] = a.mode
        ns["PROPOSAL_DETERMINISTIC"] = True
        ns["SELECTOR_SEED"] = int(a.selector_seed)
        ns["ENFORCE_UNOCCUPIED_LIGHTS_OFF"] = (a.safeguard == "on")
        ns["SYNC_PREV_WITH_DEADBAND"] = True
        ns["setup"]()                     # the callback would call this lazily on its first step
        return {"pipeline": "new_dataset_gen1 (cells 0-4, 14, 46)",
                "callback": "gen1_14_eval_callback.py", "builder": "gen1_46_rollout_builder.py",
                "policy_module": "sdar_iql_train_3_updated.py"}
    exec_source((PKG / "nb_callback.py").read_text(), str(PKG / "nb_callback.py"), ns)
    ns["ENFORCE_UNOCCUPIED_LIGHTS_OFF"] = (a.safeguard == "on")
    ns["SYNC_PREV_WITH_DEADBAND"] = True
    extra = {}
    if a.mode == "clock":
        table = json.loads(Path(a.hourly_rates).read_text())["hourly_update_rates"]
        ns["HOURLY_UPDATE_RATES"] = table          # also read by the builder's consistency check
        extra["hourly_update_rates"] = table
    ns["setup"](checkpoint_path=str(ckpt), selector_mode=a.mode, selector_seed=a.selector_seed,
                periodic_phase_seed=a.phase_seed, constant_update_rates=rates, **extra)
    return {"pipeline": "Ablation_sdar_ (cells 0-4, 5, 9; patches in sdar_eval/PATCHES.md)",
            "callback": "sdar_eval/nb_callback.py", "builder": "sdar_eval/nb_builder.py",
            "policy_module": ("sdar_eval/policy_ext.py (exact substitutions on sdar_iql_train_4_updated.load_and_act)"
                              if a.mode in ("clock", "always_prev") else "sdar_iql_train_4_updated.py")}


def build_rollout(ns: dict, a, ckpt: Path, ref: Path, rates, prefix: Path, sim_info: dict, ckpt_epoch=None):
    weather_file = None if sim_info["weather_arg"] == DEFAULT_WEATHER else sim_info["weather_epw"]
    if a.mode in GEN1_MODES:
        exec_source(load_cell("gen1_46_rollout_builder.py"), str(CELLS / "gen1_46_rollout_builder.py"), ns)
        return ns["build_agent_dataset"](
            save_npz=True, save_csv=True, prefix=str(prefix),
            reference_dataset_path=str(ref), checkpoint_path=str(ckpt), checkpoint_epoch=ckpt_epoch,
            selector_mode=a.mode, proposal_deterministic=True, selector_seed=a.selector_seed,
            safeguard_enabled=(a.safeguard == "on"), weather_file=weather_file,
        )
    ns["EVAL_SELECTOR_MODE"] = a.mode
    exec_source((PKG / "nb_builder.py").read_text(), str(PKG / "nb_builder.py"), ns)
    return ns["build_agent_dataset"](
        save_npz=True, save_csv=True, prefix=str(prefix),
        reference_dataset_path=str(ref), checkpoint_path=str(ckpt), checkpoint_epoch=ckpt_epoch,
        eval_seed=None, selector_mode=a.mode, proposal_deterministic=True,
        selector_seed=a.selector_seed, fixed_update_rates=rates,
        periodic_phase_seed=a.phase_seed, safeguard_enabled=(a.safeguard == "on"),
        weather_file=weather_file,
    )


def code_hashes():
    out = {"run_eval.py": sha256(__file__), "CELLS.json": sha256(FAITHFUL / "CELLS.json")}
    out.update({f"faithful/cells/{p.name}": sha256(p) for p in sorted(CELLS.glob("*.py"))})
    out.update({f"sdar_eval/{p.name}": sha256(p) for p in sorted(PKG.glob("*.py"))})
    for mod in ("sdar_iql_train_3_updated.py", "sdar_iql_train_4_updated.py"):
        p = PROJECT_ROOT / mod
        out[mod] = sha256(p) if p.is_file() else "missing"
    return out


def main(argv=None, before_run=None):
    """before_run(ns) is a hook for faithful/verify_replay.py only."""
    a = parse_args(argv)
    t_start = time.time()
    os.chdir(PROJECT_ROOT)                      # notebook code uses project-relative paths
    sys.path.insert(0, str(PROJECT_ROOT))       # sdar_iql_train_{3,4}_updated
    sys.path.insert(0, str(PKG))                # provenance.py

    ckpt = Path(a.checkpoint)
    ckpt = ckpt if ckpt.is_absolute() else (PROJECT_ROOT / ckpt)
    ref = Path(a.reward_reference)
    ref = ref if ref.is_absolute() else (PROJECT_ROOT / ref)
    for p in (ckpt, ref):
        if not p.is_file():
            sys.exit(f"ERROR: file not found: {p}")

    run_dir = Path(a.runs_dir) / a.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not a.overwrite:
        sys.exit(f"ERROR: {run_dir} already exists; choose another --run-name or pass --overwrite")
    (run_dir / "eplus").mkdir(parents=True, exist_ok=True)

    manifest = {
        "status": "running",
        "run_name": a.run_name,
        "command": " ".join([sys.executable] + sys.argv),
        "started": _dt.datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "args": {k: v for k, v in vars(a).items()},
        "checkpoint": {"path": str(ckpt), "sha256": sha256(ckpt)},
        "reward_reference": {"path": str(ref), "sha256": sha256(ref)},
        "hourly_rates": ({"path": str(Path(a.hourly_rates).resolve()), "sha256": sha256(a.hourly_rates)}
                         if a.hourly_rates else None),
        "evaluation_code": code_hashes(),
        "notebooks": json.loads((FAITHFUL / "CELLS.json").read_text())["notebooks"],
        "versions": versions(),
    }
    mpath = run_dir / "manifest.json"

    def write_manifest():
        manifest["updated"] = _dt.datetime.now().isoformat(timespec="seconds")
        tmp = mpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, default=str))
        tmp.replace(mpath)

    rates = a.rates
    try:
        if a.expect_checkpoint_sha and manifest["checkpoint"]["sha256"] != a.expect_checkpoint_sha:
            raise RuntimeError("checkpoint sha256 differs from --expect-checkpoint-sha")
        if a.reward_reference_sha and manifest["reward_reference"]["sha256"] != a.reward_reference_sha:
            raise RuntimeError("reward reference sha256 differs from --reward-reference-sha")

        import torch
        cfg = torch.load(ckpt, map_location="cpu", weights_only=False)
        manifest["checkpoint"].update(epoch=cfg.get("epoch"), run_name=cfg["config"].get("run_name"),
                                      dataset_path=cfg["config"].get("dataset_path"),
                                      iql_beta=cfg["config"].get("iql_beta"))
        del cfg
        if Path(manifest["checkpoint"]["dataset_path"] or "").name != ref.name:
            manifest.setdefault("warnings", []).append(
                f"checkpoint was trained on {manifest['checkpoint']['dataset_path']} but reward reference is {ref.name}")
        write_manifest()

        # ---- notebook cells 0-4: model, glazing, lighting, EnergyPlusSetup ----
        ns = {"__name__": "sdar_eval_namespace"}
        manifest["simulation"] = build_simulation(ns, a.weather, a.smoke_days)
        write_manifest()

        # ---- callback cell + agent ----
        manifest["code_path"] = prepare_agent(ns, a, ckpt, rates)
        if a.mode not in GEN1_MODES and rates is None:
            rates = dict(ns["CONSTANT_UPDATE_RATES"])   # the ablation callback's defaults
        if ns["_AGENT"]["loaded_from"] != str(ckpt):
            raise RuntimeError("agent was not loaded from the requested checkpoint")
        eps = ns["eps"]
        eps.set_callback("callback_begin_system_timestep_before_predictor", ns["callback_func"])  # cell 16
        if a.smoke_days:
            # TEST ONLY. eps.run(annual=True) forces the full weather year, so a
            # separate end-of-zone-timestep hook stops EnergyPlus after N days.
            # It is registered directly on the runtime API, leaving the agent
            # callback and frads' own callbacks untouched.
            n_stop = int(a.smoke_days) * 144

            def _smoke_stop(state):
                if len(ns["agent_data"]["observation"]) >= n_stop:
                    eps.api.runtime.stop_simulation(state)

            eps.api.runtime.callback_end_zone_timestep_after_zone_reporting(eps.state, _smoke_stop)
            print(f"[run_eval] SMOKE TEST ONLY: stopping after {a.smoke_days} days ({n_stop} steps)")
        if before_run is not None:
            before_run(ns)
        write_manifest()

        t_sim = time.time()
        eps.run(output_directory=str(run_dir / "eplus"), annual=True)                             # cell 17
        manifest["timing"] = {"simulation_s": round(time.time() - t_sim, 1)}
        first, last = energyplus_version(run_dir / "eplus")
        manifest["energyplus"] = {"version_line": first, "completion_line": last}
        if ns["_AGENT"]["loaded_from"] != str(ckpt):
            raise RuntimeError("agent checkpoint changed during the run")
        write_manifest()

        # ---- builder cell (same namespace, so it sees the callback logs) ----
        build_rollout(ns, a, ckpt, ref, rates, run_dir / "rollout", manifest["simulation"],
                      ckpt_epoch=manifest["checkpoint"].get("epoch"))

        # ---- weight-level provenance check ----
        from provenance import check_rollout
        prov = check_rollout(str(run_dir / "rollout.npz"), str(ckpt), a.provenance_steps)
        manifest["provenance"] = prov
        if not prov["match"]:
            raise RuntimeError(f"PROVENANCE FAILURE: logged logits do not match checkpoint "
                               f"({prov['mean_abs_logit_diff']:.4f})")

        manifest["outputs"] = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                               for p in sorted(run_dir.glob("rollout*")) if p.is_file()}
        manifest["status"] = "ok" if not a.smoke_days else "ok_smoke_test_not_a_result"
    except BaseException as e:  # record failures honestly, including Ctrl-C
        manifest["status"] = "failed"
        manifest["error"] = f"{type(e).__name__}: {e}"
        manifest["traceback"] = traceback.format_exc()
        write_manifest()
        print(manifest["traceback"], file=sys.stderr)
        sys.exit(1)
    finally:
        manifest["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
        manifest.setdefault("timing", {})["total_s"] = round(time.time() - t_start, 1)
        write_manifest()
    print(f"\nOK  {a.run_name}: provenance mean |dlogit| = {manifest['provenance']['mean_abs_logit_diff']:.2e}; "
          f"outputs in {run_dir}")
    return ns, manifest


if __name__ == "__main__":
    main()
