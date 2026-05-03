# Intern Challenge: Placement Problem - Fork this repo when you start!

Welcome to the par.tcl 2026 ML Sys intern challenge! Your task is to solve a placement problem involving standard cells (small blocks) and macros (large blocks). The **primary goal is to minimize overlap** between blocks. Wirelength is also evaluated, but **overlap is the dominant objective**. A valid placement must eventually ensure no blocks overlap, but we will judge solutions by how effectively you reduce overlap and, secondarily, how well you handle wirelength.

The deadline is when all intern slots for summer 2026 are filled. We will review submissions on a rolling basis.

## Problem Statement

- **Objective:** Place a set of standard cells and macros on a chip layout to **minimize overlap (most important)** and wirelength (secondary).  
  - Overlap will be measured as `num overlapping cells / num total cells`, though you are encouraged to define and implement your own overlap loss function if you think it’s better.  
  - Solving this problem will require designing a strong overlap loss, tuning hyperparameters, and experimenting with optimizers. Creativity is encouraged — nothing is off the table.  
- **Input:** Randomly generated netlists.  
- **Output:** Average normalized **overlap (primary metric)** and wirelength (secondary metric) across a set of randomized placements.  

## Submission Instructions

1. **Fork this repository.**
2. Solve the placement problem using your preferred tools or scripts.  
3. Run the first 10 tests to evaluate your solution and obtain the overlap and wirelength metrics. Report Average Overlap, Wirelength and total Runtime. *Test cases 11 and 12 are extra credit, give them a shot if you have some time.*  
5. Submit a pull request with your updated leaderboard entry and instructions for me to access your actual submission (it's fine if it's public).

Note: You can use any libraries or frameworks you like, but please ensure that your code is well-documented and easy to follow.  

Also, if you think there are any bugs in the provided code, feel free to fix them and mention the changes in your submission.  

You may submit multiple solutions to try and increase your score.

We will review submissions on a rolling basis.

## Leaderboard (sorted by overlap)

| Rank | Name            | Overlap     | Wirelength (um) | Runtime (s) | Notes                |
|------|-----------------|-------------|-----------------|-------------|----------------------|
| 1    | Brayden Rudisill  | 0.0000    | 0.2611          |   50.51     |   Timed on a mac air |
| 2    | manuhalapeth      | 0.0000    | 0.2630          |  196.8      |                      |
| 3    | Neil Teje         | 0.0000    | 0.2700          | 24.00s      |                      |
| 4    | Leison Gao      | 0.0000      | 0.2796          | 50.14s      |                      |
| 5    | William Pan     | 0.0000      | 0.2848          | 155.33s     |                      |
| 6    | Ashmit Dutta    | 0.0000      | 0.2870          | 995.58      |  Spent my entire morning (12 am - 6 am) doing this :P       |
| 7    | Pawan Paleja     | 0.0000      | 0.3311         | 1.74s     |   Implemented hint for loss func, cosine annealing on learning rate with warmup, std annealing on lambda weight. Used optuna to tune hyperparam. Tested on gh codespaces 2-core.                   |
 8   | Shashank Shriram  | 0.0000     | 0.3312          |  11.32      |   🏎️💥               |
| 9    | Gabriel Del Monte  | 0.0000      | 0.3427          | 606.07      |                                                              |
| 10    | Aleksey  Valouev| 0.0000      | 0.3577          | 118.98      |                      |        
| 11   | Mohul Shukla    | 0.0000      | 0.5048          | 54.60s      |                      |
| 12    | Ryan Hulke      | 0.0000      | 0.5226          | 166.24      |                      |
| 13    | Neel  Shah      | 0.0000      | 0.5445          | 45.40       |  Zero overlaps on all tests, adaptive schedule + early stop |
| 14   | Nawel Asgar    | 0.0000     | 0.5675          | 81.49      | Adaptive penalty scaling with cubic gradients and design-size optimization
| 15   | Shiva Baghel     | 0.0000     | 0.5885          | 491.00      | Stable zero-overlap with balanced optimization      |
| 16   | Vansh Jain      | 0.0000      | 0.9352          | 86.36       |                      |
| 17    | Akash Pai       | 0.0006      | 0.4933          | 326.25s     |                      |
| 18    | Zade Mahayni     | 0.00665     | 0.5157          |  127.4     | Will try again tomorrow |
| 19    | Nithin Yanna    | 0.0148      | 0.5034          | 247.30s     | aggressive overlap penalty with quadratic scaling |
| 20    | Sean Ko         | 0.0271      |  .5138          | 31.83s      | lr increase, decrease epoch, increase lambda overlap and decreased lambda wire_length + log penalty loss |
| 21    | Keya Gohil    | 0.0155      | 0.4678         | 1513.07     | Still working |
| 22    | Prithvi Seran   | 0.0499      | 0.4890          | 398.58      |                      |
| 23    | partcl example  | 0.8         | 0.4             | 5           | example              |
| 24    | Add Yours!      |             |                 |             |                      |

> **To add your results:**  
> Insert a new row in the table above with your name, overlap, wirelength, and any notes. Ensure you sort by overlap.

Good luck!

## Repo Orientation

This fork is no longer a single flat "try anything" challenge repo. It now has
four practical perspectives:

1. **Training**
   - Start at `train_ppo.py`
   - Then read `ppo.py`, `ordering_policy.py`, and `env.py`
2. **Environment / geometry transition**
   - Start at `env.py`
   - Then read `constraints.py`, `active_set.py`, `primal_dual.py`, and
     `induce_branches.py`
3. **Inference / deployment**
   - Start at `placement.py`
   - Then read `validate_policy.py` and `compare_inference_modes.py`
4. **Tracing / debugging / analysis**
   - Start at `run_diagnosis_suite.py`
   - Then read `run_decision_tests.py`, `counterfactual_replay.py`, `visualize_rollout_trace.py`, `validity_tests.py`, `test.py`, `runs/`, and `traces/`

If you only want the shortest possible route through the repo:

- **I want to train the policy**
  - Read [`docs/TRAINING.md`](docs/TRAINING.md)
- **I want to understand the environment transition**
  - Read [`docs/REPO_MAP.md`](docs/REPO_MAP.md) and then `env.py`
- **I want to call the final inference API**
  - Read [`docs/INFERENCE_AND_DEPLOYMENT.md`](docs/INFERENCE_AND_DEPLOYMENT.md)
- **I want to inspect rollout behavior**
  - Read [`docs/TRACE_AND_DEBUGGING.md`](docs/TRACE_AND_DEBUGGING.md)

There are also a few intentionally thin compatibility facades:

- `policy.py` re-exports the real policy implementation from `ordering_policy.py`
- `ordering.py` re-exports ordering helpers from `ordering_policy.py` and
  `induce_branches.py`
- `rollout.py` re-exports environment-facing rollout helpers from `env.py`
- `prior_solver.py` is a thin offline teacher alias around `teacher_solver.py`

Those are not the best places to start reading the code.

## Policy-Conditioned PPO Implementation

This fork includes a policy-conditioned primal-dual PPO path and an offline
teacher-distillation path:

- `train_ppo.py` trains the structured PPO policy. Add `--teacher-dataset path.pt`
  with `--distill-epochs N` to run integrated outcome distillation before PPO
  begins, then anneal an offline teacher auxiliary loss during early PPO
  updates via `--teacher-anneal-updates`; the training log carries
  teacher-source, demo-quality, and live `lambda_teacher` provenance.
- `teacher_data.py` builds offline outcome datasets from a prior solver. By
  default it points at `prior_solver:train_prior_solver_placement` and disables
  policy inference while the prior solver is running, so the teacher remains an
  offline demonstration source.
- `distill.py` performs confidence-weighted outcome distillation from saved
  teacher data. It is still useful for producing a standalone warm-start
  checkpoint, but the paper-aligned default is the integrated `train_ppo.py`
  path so teacher provenance is preserved in the final PPO logs.
- `train_mac.sh` is the paper-aligned default workflow for Apple Silicon. It
  generates the offline teacher dataset, runs integrated PPO warm-start
  distillation with metric-gated hardening, validates the trained checkpoint,
  reports declared Mode A vs Mode B inference behavior, and runs the lightweight
  validity suite.
- `validate_policy.py` now evaluates declared inference modes explicitly via
  `--mode terminal_policy`, `--mode audited_policy_ensemble`, or `--mode both`.
- `compare_inference_modes.py` reports the gap between terminal policy mode and
  audited policy ensemble mode using official lexicographic metric selection.
- `placement.train_placement(...)` now exposes the declared inference contract
  directly through `mode`, `num_rollouts`, `num_steps`, and `temperature`
  instead of hiding Mode A / Mode B selection behind environment variables.
- `validity_tests.py` runs lightweight validity checks for exact audit parity,
  branch antisymmetry, no-teacher inference, Mode B no-repair selection, stop
  safety, teacher annealing, active-set scale, and distillation smoke coverage.

## Reading Order

### If you are changing training behavior

Read in this order:

1. `train_ppo.py`
2. `ppo.py`
3. `ordering_policy.py`
4. `env.py`

### If you are changing the placement transition itself

Read in this order:

1. `env.py`
2. `constraints.py`
3. `active_set.py`
4. `primal_dual.py`
5. `induce_branches.py`

### If you are changing inference or benchmark-facing behavior

Read in this order:

1. `placement.py`
2. `validate_policy.py`
3. `compare_inference_modes.py`

### If you are changing teacher warm-starts

Read in this order:

1. `teacher_data.py`
2. `distill.py`
3. `teacher_solver.py`
4. `prior_solver.py`

### If you are debugging rollout behavior

Read in this order:

1. `run_diagnosis_suite.py`
2. `run_decision_tests.py`
3. `counterfactual_replay.py`
4. `visualize_rollout_trace.py`
5. `env.py`
6. `ordering_policy.py`
7. `validity_tests.py`

## Extra Documentation

- [`docs/REPO_MAP.md`](docs/REPO_MAP.md): repo grouped by system boundary
- [`docs/TRAINING.md`](docs/TRAINING.md): training workflow and key files
- [`docs/INFERENCE_AND_DEPLOYMENT.md`](docs/INFERENCE_AND_DEPLOYMENT.md):
  inference wrapper and deployment-facing contract
- [`docs/TRACE_AND_DEBUGGING.md`](docs/TRACE_AND_DEBUGGING.md): trace outputs,
  rollout diagnostics, and debugging surfaces
- [`docs/DIAGNOSIS_PLAYBOOK.md`](docs/DIAGNOSIS_PLAYBOOK.md): persistent
  analysis breadcrumbs, dominant-factor interpretation rules, and next-step
  guidance after diagnosis runs
- [`docs/EXPERIMENT_DECISION_MEMO.md`](docs/EXPERIMENT_DECISION_MEMO.md):
  current experiment priorities, what not to touch yet, and falsification rules
