# Verification: `run_eval.py` reproduces the 7 Aug beta3 epoch-100 rollout

**Date:** 1 Oct 2026. **Machine:** a CPU-only machine (2 CPUs, no GPU), with the same library versions as `offrl_5zone`:

| Library | Version |
|---|---|
| frads | 2.1.15 |
| pyenergyplus_lbnl (EnergyPlus) | 25.2.0 |
| pyradiance | 1.1.5 |
| torch | 2.11 |

**What I ran:**

```bash
python aamas2027/scripts/faithful/verify_replay.py --days 0 \
  --reference agent_eval_dataset_decomposed_epoch500_thermal3_tclip50_selector-sample_propdet_s20260728.npz \
  --checkpoint checkpoints/<beta3>/..._checkpoint_epoch100.pt
```

This is the full year: notebook cells 0–4, 14 and 46, executed verbatim. The selector's Bernoulli draws were replaced by the masks that the reference rollout logged. The notebook drew those masks on CUDA, so a CPU run cannot regenerate them from the seed.

## Annual results: reference vs. re-run

| KPI | 7 Aug notebook run | Re-run with `run_eval.py` | Difference |
|---|---:|---:|---:|
| Combined electricity (kWh/day) | 345.378 | 345.342 | −0.036 |
| HVAC (kWh/day) | 325.296 | 325.261 | −0.035 |
| Lighting (kWh/day) | 20.082 | 20.081 | −0.001 |
| Occupied thermal violations (%) | 7.475 | 7.516 | +0.041 |
| Severity (K·zone·h/day) | 1.235 | 1.243 | +0.008 |
| Perimeter visual in-band (%) | 48.624 | 48.613 | −0.011 |
| Daylight > 1000 lx (%) | 24.326 | 24.309 | −0.017 |
| Update rate (%), and by subsystem | 13.603 (1.769 / 10.817 / 19.639 / 19.818) | identical | 0 |
| Clearest glazing tint (% of occupied window-steps) | 33.359 | 33.359 | 0 |

## Step by step (52,559 steps)

- **Exact match.** Weather-driven inputs (transmitted solar and exterior irradiance), the time features and every update mask are identical. Glazing actions are identical in 99.97 % of steps.
- **Zone temperature.** The median difference is 0.0001 °C. Differences above 0.1 °C occur in 0.74 % of steps, in short episodes (for example on days 11, 91 and 220) that die out again.
- **Heating and cooling setpoints.** The median difference is 0.0003 (normalised units). It exceeds 0.01 in about 2.5 % of steps.

**Cause.** The residual differences come from floating-point differences between CPU and GPU in the GRU and the Gaussian proposal heads. They become briefly visible when a setpoint lands near the heating/cooling deadband or a comfort threshold. They do not accumulate, and they change the annual KPIs only in the second decimal place.

The **provenance check** on the re-run gave a mean |Δlogit| of 3.1e-4, well inside the match tolerance of 1e-2.

**Conclusion.** The runner reproduces the beta3 epoch-100 result that the paper reports. On the workstation's GPU it should reproduce the reference more closely still. Run `verify_replay.py --days 3 --no-replay-masks` there to check the random draws as well.
