# Repo Map

This repo is easiest to understand if you stop reading it as a flat list of
Python files and instead read it by **system boundary**.

## 1. Training

Use this slice when you are asking:

- How are rollouts collected?
- How is PPO updated?
- How are validation, hardening, replay, and checkpointing handled?

Start here:

1. `train_ppo.py`
2. `ppo.py`
3. `ordering_policy.py`
4. `env.py`

Core files:

- `train_ppo.py`
  - main single-process trainer
  - fixed validation suite
  - hard replay suite
  - checkpoint selection
- `ppo.py`
  - PPO loss
  - GAE / return handling
  - action-group accounting
- `ordering_policy.py`
  - actor / critic model
  - ordering heads
  - PD-control heads
  - residual-flow heads
- `train_ppo_sync.py`
  - alternate training path
- `train_multi_gpu.py`
  - multi-device launcher
- `train_ppo_multi_gpu.py`
  - older multi-GPU trainer variant
- `ablate_ppo.py`
  - targeted ablation entrypoint
- `run_ppo_ablations.py`
  - ablation launcher

## 2. Environment and Geometry Transition

Use this slice when you are asking:

- What does one placement step actually do?
- Where is the policy-conditioned primal-dual transition?
- How are active pairs, audits, and duals updated?

Start here:

1. `env.py`
2. `constraints.py`
3. `active_set.py`
4. `primal_dual.py`
5. `induce_branches.py`

Core files:

- `env.py`
  - `PlacementOrderingEnv`
  - rollout transition
  - reward calculation
  - coordinate layer
  - dual updates
  - active-set audit
- `constraints.py`
  - overlap geometry
  - signed branch constraints
  - density / boundary helpers
- `active_set.py`
  - active-pair construction and retention
- `primal_dual.py`
  - PHR update helpers
- `induce_branches.py`
  - branch induction from sequence pairs

## 3. Inference and Deployment

Use this slice when you are asking:

- What is the public placement API?
- What does benchmark-facing inference do?
- How do terminal policy mode and audited ensemble mode differ?

Start here:

1. `placement.py`
2. `validate_policy.py`
3. `compare_inference_modes.py`

Core files:

- `placement.py`
  - `train_placement(...)`
  - deployment-facing wrapper
  - inference mode handling
- `validate_policy.py`
  - deterministic evaluation runs
- `compare_inference_modes.py`
  - Mode A vs Mode B comparison

## 4. Teacher / Distillation

Use this slice when you are asking:

- How is offline teacher data built?
- How is outcome distillation run?
- Where is the teacher boundary enforced?

Start here:

1. `teacher_data.py`
2. `distill.py`
3. `teacher_solver.py`
4. `prior_solver.py`

Core files:

- `teacher_data.py`
  - offline dataset creation
- `distill.py`
  - outcome-only distillation
- `teacher_solver.py`
  - deterministic teacher implementation
- `prior_solver.py`
  - thin alias used to keep the teacher clearly offline

## 5. Trace, Validation, and Debugging

Use this slice when you are asking:

- What is the policy actually doing step by step?
- How do I inspect overlaps, duals, or inner PHR motion?
- What sanity checks exist?

Start here:

1. `visualize_rollout_trace.py`
2. `validity_tests.py`
3. `test.py`
4. `runs/`
5. `traces/`

Core files:

- `visualize_rollout_trace.py`
  - comprehensive rollout tracer
  - frame rendering
  - inner-PHR trace
  - active-pair / dual dumps
  - ordering / logit dumps
  - multi-rollout ensemble traces
- `validity_tests.py`
  - lightweight implementation validity checks
- `test.py`
  - challenge evaluation cases and simple evaluation path
- `runs/`
  - training run artifacts
- `traces/`
  - rollout trace artifacts

## 6. Thin Facades

These files exist mostly for naming compatibility. They are not the best place
to start if you want the real implementation.

- `policy.py`
- `ordering.py`
- `rollout.py`
- `prior_solver.py`

## 7. Practical Reading Paths

### I need to change reward behavior

Read:

1. `train_ppo.py`
2. `env.py`
3. `ppo.py`

### I need to change ordering behavior

Read:

1. `ordering_policy.py`
2. `induce_branches.py`
3. `env.py`

### I need to change active-set / legality behavior

Read:

1. `active_set.py`
2. `constraints.py`
3. `env.py`

### I need to change deployment behavior

Read:

1. `placement.py`
2. `validate_policy.py`

