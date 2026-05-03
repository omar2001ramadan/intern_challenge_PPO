# Trace and Debugging

The main debugging surface for policy behavior is:

- `run_diagnosis_suite.py`
- `run_decision_tests.py`
- `visualize_rollout_trace.py`
- `counterfactual_replay.py`

## Recommended starting point

If the question is "what should we fix next?", start with:

- `run_diagnosis_suite.py`

That runner:

1. reconstructs the fixed validation suite for a checkpoint
2. ranks validation cases individually
3. runs counterfactual replays on the worst cases
4. emits a diagnosis report pointing at the likely dominant factor

Typical usage:

```bash
python3 /Users/omarramadan/Desktop/compression/intern_challenge_PPO/run_diagnosis_suite.py \
  --checkpoint /absolute/path/to/checkpoint.pt \
  --output-dir /absolute/path/to/diagnosis_dir
```

Useful outputs:

- `validation_case_ranking.csv`
- `case_diagnosis.csv`
- `next_steps.json`
- `diagnosis_summary.json`
- `decision_test_results.json`
- `decision_test_report.md`
- `README.md`

Each analyzed case then gets its own counterfactual subdirectory.

For the persistent interpretation rules and future-work breadcrumbs, read:

- [`docs/DIAGNOSIS_PLAYBOOK.md`](/Users/omarramadan/Desktop/compression/intern_challenge_PPO/docs/DIAGNOSIS_PLAYBOOK.md)
- [`docs/EXPERIMENT_DECISION_MEMO.md`](/Users/omarramadan/Desktop/compression/intern_challenge_PPO/docs/EXPERIMENT_DECISION_MEMO.md)

If the diagnosis suite already exists and the question is "does the current
interpretation still hold?", run:

```bash
python3 /Users/omarramadan/Desktop/compression/intern_challenge_PPO/run_decision_tests.py \
  --diagnosis-dir /absolute/path/to/diagnosis_dir
```

## What the tracer gives you

Per rollout:

- `steps.csv`
- `steps.jsonl`
- `coordinates.csv`
- `overlap_pairs.csv`
- `phr_steps.csv`
- `phr_coordinates.csv`
- `active_pair_duals.csv`
- `boundary_duals.csv`
- `density_duals.csv`
- `ordering_scores.csv`
- `cluster_logits.csv`
- `pair_branch_logits.csv`
- `policy_trace.jsonl`
- `timeline.png`
- `frames/`

For multi-rollout runs on the same case:

- `ensemble_candidates.csv`
- `ensemble_summary.json`

For one-factor-at-a-time counterfactual analysis on one case:

- `counterfactual_replay.py`

Useful outputs:

- `variant_summary.csv`
- `all_variant_steps.csv`
- `all_action_deltas.csv`
- `memory_transition_invariance.csv`

## Debugging questions this answers well

### What did the policy propose?

Use:

- `coordinates.csv`
- `policy_trace.jsonl`
- `ordering_scores.csv`

### What did the inner PHR layer do?

Use:

- `phr_steps.csv`
- `phr_coordinates.csv`

### Which active pairs and duals were carrying pressure?

Use:

- `active_pair_duals.csv`
- `boundary_duals.csv`
- `density_duals.csv`

### Did multiple rollout samples land in different basins?

Use:

- `ensemble_candidates.csv`
- per-rollout subdirectories

## Other debugging files

- `validity_tests.py`
  - implementation sanity checks
- `test.py`
  - challenge cases
- `runs/`
  - training logs and checkpoints
- `traces/`
  - generated trace artifacts

## Typical debugging flow

1. Run a deterministic or low-noise trace on one fixed case.
2. Inspect `timeline.png` and `steps.csv`.
3. Inspect `phr_steps.csv` to see whether inner optimization is actually moving.
4. Inspect `active_pair_duals.csv` to see whether the system knows where the conflicts are.
5. Inspect `ordering_scores.csv` and `policy_trace.jsonl` to see what the policy believed.
6. If needed, rerun with `--num-rollouts > 1` to compare multiple sampled basins on the same case.
