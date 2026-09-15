# configs/

Sweep-configuration JSON files, loaded by `scripts/generate_dataset.py --sweep-config <file>` (each key maps directly onto a `network.sweep.SweepConfig` field — see that module for validation rules; JSON has no comment syntax, so explanations live here instead of inline).

## pilot_sweep.json

Phase 5's pilot sweep (`docs/PHASE_5_REAL_DATA_VALIDATION.md` Section 5). Deliberately small (12 experiments: 2 bandwidths × 2 delays × 3 offered-load factors × 1 flow count × 1 repetition, ~4 minutes of experiment time at 20s each) but spans **below** (0.5×), **at** (1.0×), and **above** (1.5×) bottleneck bandwidth — the "above" condition is what should produce genuine, observable, queue-overflow-driven packet loss (see `docs/PHASE_1_NETWORK_EXPERIMENT.md` Section 6). `n_flows_values` stays `[1]`: Phase 1's experiment runner only drives a single sender/receiver pair (documented limitation), so this dimension can't be widened without first extending `src/network/experiment.py`.

Run:
```bash
python3 scripts/generate_dataset.py --sweep-config configs/pilot_sweep.json --dry-run   # verify the plan, no network touched
python3 scripts/generate_dataset.py --sweep-config configs/pilot_sweep.json             # real run, requires Mininet
```
