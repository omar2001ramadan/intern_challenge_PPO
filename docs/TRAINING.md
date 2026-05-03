# Training Workflow

This is the file path to follow if your job is “make the policy train better.”

## Primary entrypoint

- `train_ppo.py`

That file owns:

- CLI arguments
- rollout collection
- validation scheduling
- soft-to-hard gating
- hard replay construction
- checkpoint selection
- logging

## Training stack

Read in this order:

1. `train_ppo.py`
2. `ppo.py`
3. `ordering_policy.py`
4. `env.py`

## Mental model

`train_ppo.py` does not define the geometry. It orchestrates:

- problem generation
- policy rollout
- PPO update
- deterministic validation
- checkpoint bookkeeping

The actual one-step placement transition still lives in `env.py`.

## Files by role

### Trainer orchestration

- `train_ppo.py`
- `train_ppo_sync.py`
- `train_multi_gpu.py`
- `train_ppo_multi_gpu.py`

### PPO update logic

- `ppo.py`

### Policy network

- `ordering_policy.py`
- `policy.py` is only a facade

### Offline warm-start / teacher path

- `teacher_data.py`
- `distill.py`
- `teacher_solver.py`
- `prior_solver.py`

## Common modification points

### Reward or validation drift

Look at:

1. `env.py`
2. `train_ppo.py`

### PPO instability

Look at:

1. `ppo.py`
2. `ordering_policy.py`

### Wrong hardening behavior

Look at:

1. `train_ppo.py`
2. `env.py`

### Training cases are too easy / too random

Look at:

1. `train_ppo.py`
2. `placement.generate_placement_input`

## Recommended training commands

### Main trainer

```bash
python3 train_ppo.py --help
```

### Validate a checkpoint

```bash
python3 validate_policy.py --help
```

### Build teacher data

```bash
python3 teacher_data.py --help
```

### Run standalone distillation

```bash
python3 distill.py --help
```

