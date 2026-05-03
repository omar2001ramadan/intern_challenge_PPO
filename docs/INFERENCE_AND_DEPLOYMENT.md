# Inference and Deployment

This repo’s deployment-facing contract is the placement wrapper in
`placement.py`.

## Public API

Start here:

- `placement.py`

The main public function is:

- `train_placement(cell_features, pin_features, edge_list, ...)`

Despite the name, this is the benchmark-facing inference wrapper. It loads the
policy-conditioned transition path and returns final cell centers.

## Files that matter

### Core deployment path

- `placement.py`

### Evaluation helpers

- `validate_policy.py`
- `compare_inference_modes.py`
- `test.py`

## Inference modes

The wrapper exposes two declared inference modes:

- `terminal_policy`
  - one rollout, return the policy terminal state
- `audited_policy_ensemble`
  - multiple rollout candidates, select by exact overlap then wirelength

If you are reasoning about benchmark behavior, this distinction matters.

## Where deployment behavior is defined

### Candidate generation

- `placement.py`
- `env.py`
- `ordering_policy.py`

### Candidate selection

- `placement.py`
- `compare_inference_modes.py`

### Exact metric evaluation

- `placement.py`
- `constraints.py`
- `rollout.py`

## If you need to modify inference behavior

Read in this order:

1. `placement.py`
2. `env.py`
3. `ordering_policy.py`
4. `validate_policy.py`

## If you need to answer “what gets shipped?”

The shortest answer is:

- `placement.py` is the deployment wrapper
- `train_placement(...)` is the public API
- `validate_policy.py` is the deterministic evaluation entrypoint

