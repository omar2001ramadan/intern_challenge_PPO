# Experiment Decision Memo

Date: 2026-04-26

This memo converts the current diagnosis into a concrete experimental plan.

Primary diagnosis source:

- [diagnosis_suite_best_exact_full](/Users/omarramadan/Desktop/compression/intern_challenge_PPO/traces/diagnosis_suite_best_exact_full)
- [case_diagnosis.csv](/Users/omarramadan/Desktop/compression/intern_challenge_PPO/traces/diagnosis_suite_best_exact_full/case_diagnosis.csv)
- [next_steps.json](/Users/omarramadan/Desktop/compression/intern_challenge_PPO/traces/diagnosis_suite_best_exact_full/next_steps.json)

Checkpoint under analysis:

- [best_exact_overlap.pt](/Users/omarramadan/Desktop/compression/intern_challenge_PPO/runs/recovery_mvp_rewardfix_live_20260426_101536/ppo/best_exact_overlap.pt)

## Current Interpretation

The current reward-fix checkpoint is not primarily failing because of:

- active-set incompleteness
- insufficient PHR budget

The two strongest remaining factors are:

1. continuation after a good basin
2. recurrent-memory sensitivity

The case split is:

- continuation-dominated:
  - `2:30 seed 1001236`
  - `3:50 seed 1001237`
- memory-sensitive:
  - `2:20 seed 1001234`
  - `3:25 seed 1001235`

## What To Test First

### 1. Best-So-Far Preservation / Horizon / Stop

Question:

- Does the policy find a good basin and then damage it because rollout continues?

Why this is first:

- On the worst `2:30` case, cutoff `2` preserves a much better state than full horizon `8`.
- On the worst `3:50` case, cutoff `6` preserves a better state than full horizon `8`.
- Later degrading steps still receive positive reward.

Immediate test:

- Re-run the diagnosis suite on the checkpoint.
- Run decision tests on the diagnosis directory.
- Inspect baseline vs best-cutoff traces for the continuation-dominated cases.

Interpretation target:

- We expect best-so-far overlap to appear earlier than the final returned state on multiple hard cases.

### 2. Recurrent-Memory Stability

Question:

- Does carried recurrent state perturb continuous controls early and then alter ordering later?

Why this is second:

- In memory-sensitive cases, the first real divergence appears at step `1`.
- The first divergence is not an immediate ordering flip.
- The first divergence is in residual flow and PD controls.
- Ordering divergence follows later.

Immediate test:

- Compare `carry`, `zero_each_step`, and `freeze_initial`.
- Read `all_action_deltas.csv`, `memory_summary.csv`, and `all_variant_steps.csv`.
- Identify whether memory helps or hurts on each case.

Interpretation target:

- We expect memory to be high-leverage but not uniformly good or bad.

## What Not To Touch Yet

Do not spend the next iteration on these first:

1. Active-set redesign
   - `active_set_all_pairs` did not materially improve legality on the fixed suite.

2. PHR budget expansion or inner-optimizer rewrites
   - the `phr_pd_steps_*` ladder did not produce the main legality gains.

3. Re-enabling cluster hierarchy
   - the causal core is still unresolved; hierarchy adds policy complexity before the current failure mode is fixed.

4. Re-enabling teacher auxiliary loss
   - current questions are about online rollout behavior, not warm-start quality.

5. Declaring a single fixed short horizon as the solution
   - cutoff `2` helps one hard case, cutoff `6` helps another.
   - this points to stop / preserve behavior, not one global magic cutoff.

## Falsification Criteria

The current interpretation should be treated as wrong or incomplete if one of these happens on later checkpoints:

### Falsifier A: Active Set Is Actually Primary

If `active_set_all_pairs` improves final overlap by at least `0.03` on at least `2` fixed-suite cases, active-set coverage becomes a first-order candidate.

### Falsifier B: PHR Budget Is Actually Primary

If some `phr_pd_steps_*` variant improves final overlap by at least `0.03` on at least `2` fixed-suite cases, PHR budget sensitivity becomes first-order.

### Falsifier C: Continuation Is Not Really The Main Failure

If shorter cutoffs do not materially improve final legality on the hard cases, or if the final returned state is already near the best-so-far state, then the continuation story weakens.

### Falsifier D: Memory Is Not Causal Enough To Matter

If memory variants no longer produce material action deltas or overlap deltas on the same cases, recurrent-state handling is no longer a priority.

## Current Decision

The next engineering direction should be judged by these priorities:

1. preserve the good basin once found
2. stabilize or constrain recurrent-state influence on the control path
3. only then reconsider active-set or PHR internals if the falsifiers trigger

## Commands

Rebuild the diagnosis suite:

```bash
python3 /Users/omarramadan/Desktop/compression/intern_challenge_PPO/run_diagnosis_suite.py \
  --checkpoint /Users/omarramadan/Desktop/compression/intern_challenge_PPO/runs/recovery_mvp_rewardfix_live_20260426_101536/ppo/best_exact_overlap.pt \
  --device cpu \
  --top-k-cases 4 \
  --output-dir /Users/omarramadan/Desktop/compression/intern_challenge_PPO/traces/diagnosis_suite_best_exact_full
```

Evaluate the decision tests on that diagnosis directory:

```bash
python3 /Users/omarramadan/Desktop/compression/intern_challenge_PPO/run_decision_tests.py \
  --diagnosis-dir /Users/omarramadan/Desktop/compression/intern_challenge_PPO/traces/diagnosis_suite_best_exact_full
```
