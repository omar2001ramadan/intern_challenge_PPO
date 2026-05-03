# Diagnosis Playbook

This document is the durable breadcrumb trail for future debugging.

If the repo is in a confusing state later, start here and then run:

```bash
python3 /Users/omarramadan/Desktop/compression/intern_challenge_PPO/run_diagnosis_suite.py \
  --checkpoint /absolute/path/to/checkpoint.pt \
  --output-dir /absolute/path/to/diagnosis_dir
```

That command will emit:

- `validation_case_ranking.csv`
- `case_diagnosis.csv`
- `next_steps.json`
- `diagnosis_summary.json`
- per-case `case_next_steps.json`

The intent is:

1. rank the fixed validation cases individually
2. run counterfactual replays on the worst cases
3. decide which factor is most likely causal
4. preserve the recommended next experiments in machine-readable form

## Ground Rules

These rules are here so future analysis does not drift:

1. Do not trust aggregate training means when per-case counterfactuals disagree.
2. Do not treat memory as a transition-side factor under frozen actions.
3. Do not blame active-set coverage unless `active_set_all_pairs` materially improves legality.
4. Do not blame PHR budget unless the `phr_pd_steps_*` ladder materially changes legality.
5. If a short horizon preserves legality while the full horizon degrades, continuation is the primary failure mode until disproven.

## Dominant Factors

### `continuation_after_good_basin`

Meaning:

- The rollout finds a good basin early.
- Later steps degrade legality or reintroduce overlap.

What to change next:

- Investigate stop behavior, horizon length, and best-so-far preservation.
- Compare early cutoff traces against the full rollout on the same case.

Evidence pattern:

- `horizon_cutoff_k` materially improves final legality.
- Best-so-far appears early and then stays flat.
- `active_set_all_pairs` and `phr_pd_steps_*` do not materially change legality.

Do not do this:

- Do not add more rollout steps by default.
- Do not start by rewriting PHR internals if cutoff-2 already preserves the good basin.

### `active_set_incompleteness`

Meaning:

- Full all-pairs exposure improves legality over the baseline cached active set.

What to change next:

- Increase exact-audit coverage.
- Increase pair retention or exposure frequency.
- Recheck hardening gates against active-set completeness.

Evidence pattern:

- `active_set_all_pairs` materially improves overlap or overlap-pair count.

Do not do this:

- Do not blame the policy first if the transition is optimizing around hidden constraints.

### `phr_budget_sensitivity`

Meaning:

- Inner primal-dual budget materially changes legality on the same action sequence.

What to change next:

- Inspect `phr_steps.csv`.
- Tune `pd_steps`, `rho`, `alpha`, or how those controls are learned.

Evidence pattern:

- Some `phr_pd_steps_*` variants materially improve legality.

Do not do this:

- Do not treat wirelength-only differences as legality fixes.

### `memory_sensitivity`

Meaning:

- Seed-locked memory variants change sampled actions enough to alter case behavior.

What to change next:

- Inspect `all_action_deltas.csv`.
- Compare baseline and memory-variant traces at the first divergence step.
- Test whether memory changes best-so-far quality or only late-step drift.

Evidence pattern:

- `memory_zero_each_step` or `memory_freeze_initial` diverges materially from baseline.

Do not do this:

- Do not interpret memory as causal under action-locked replay.

## What To Preserve

If you run a diagnosis sweep that matters, keep these artifacts together:

- the checkpoint path
- `validation_case_ranking.csv`
- `case_diagnosis.csv`
- `next_steps.json`
- the per-case counterfactual directories

If a future run disagrees with the current diagnosis, compare those per-case directories first.
