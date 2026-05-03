"""Run an end-to-end diagnosis suite for fixed-validation PPO behavior.

This is the orchestration layer on top of:
- `validity_tests.py`
- per-case deterministic validation scoring
- `counterfactual_replay.py`

The intent is one command that:
1. reconstructs the fixed validation suite
2. ranks validation cases individually
3. runs counterfactual sweeps on the worst few cases
4. emits a report describing which factor appears causal:
   - continuation / horizon
   - active-set completeness
   - PHR budget sensitivity
   - recurrent-memory sensitivity
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

from ordering_policy import DISCOVER_MODE_NAMES, load_policy_checkpoint
from structural_audit import write_structural_outputs
from train_ppo import build_validation_suite, collect_episode, parse_sizes
from visualize_rollout_trace import env_config_from_checkpoint
from wisdom_audit import write_wisdom_outputs


DEFAULT_SIZES = "2:20,3:25,2:30,3:50,4:75,5:100,5:150"


FACTOR_PLAYBOOK = {
    "continuation_after_good_basin": {
        "label": "Continuation After Good Basin",
        "what_it_means": "The policy finds a materially better basin early, but continued rollout degrades legality or reintroduces overlap.",
        "what_to_change_next": [
            "Add or strengthen best-so-far preservation in training analysis and checkpoint review.",
            "Revisit stop / horizon behavior before reworking geometry or active-set logic.",
            "Compare fixed short-horizon validation against full-horizon validation before changing PHR internals.",
        ],
        "experiments": [
            "Run counterfactual horizon cutoffs on the worst fixed-suite cases and identify the earliest stable cutoff.",
            "Trace the baseline rollout and the best cutoff side by side to see which step first degrades post-step legality.",
            "Evaluate whether a learned stop policy or fixed short horizon would preserve the best-so-far candidate more reliably.",
        ],
        "success_criteria": [
            "Shorter-horizon or stop-aware runs preserve the early best basin on multiple fixed-suite cases.",
            "Best-so-far overlap no longer appears substantially earlier than the final returned state.",
        ],
        "anti_patterns": [
            "Do not blame PHR budget first if cutoff-2 already matches the best-so-far basin.",
            "Do not add more rollout steps by default when later steps already worsen legality.",
        ],
    },
    "active_set_incompleteness": {
        "label": "Active-Set Incompleteness",
        "what_it_means": "The transition improves when full all-pairs visibility is forced, which implies the policy or PHR layer is optimizing around missing constraints.",
        "what_to_change_next": [
            "Increase exact-audit exposure or active-pair retention before changing policy architecture.",
            "Stress-test active-set coverage on small cases with all-pairs replay against cached-pair replay.",
            "Treat missed-pair coverage as a gating metric for hardening decisions.",
        ],
        "experiments": [
            "Replay the same action sequence with baseline active set and all-pairs active set.",
            "Trace pair-lifecycle and dual growth on the same case to see which conflicts are absent too long.",
            "Raise audit pressure and retention horizon in a controlled ablation on the same fixed suite.",
        ],
        "success_criteria": [
            "All-pairs replay no longer materially improves legality over the cached active set.",
            "Missed-pair counts and inactive exact-overlap burden stay low on the fixed suite.",
        ],
        "anti_patterns": [
            "Do not conclude the policy is the main problem until all-pairs replay stops helping.",
        ],
    },
    "phr_budget_sensitivity": {
        "label": "PHR Budget Sensitivity",
        "what_it_means": "Legality depends materially on inner primal-dual budget or control settings, so the rollout proposal is not the only bottleneck.",
        "what_to_change_next": [
            "Examine whether higher PHR steps improve legality without changing early basin quality.",
            "Inspect per-inner-step traces before rewriting reward logic.",
            "Treat `pd_steps`, `rho`, and `alpha` as first-class diagnostics on failing cases.",
        ],
        "experiments": [
            "Run the same action sequence across a wide `pd_steps` ladder.",
            "Compare `phr_steps.csv` for the lowest and highest useful budgets.",
            "Check whether additional PHR budget reduces exact-overlap pairs or only changes wirelength.",
        ],
        "success_criteria": [
            "Increasing PHR budget no longer yields material legality gains on the same fixed cases.",
            "Inner-step traces show either convergence or a clear plateau rather than under-optimization.",
        ],
        "anti_patterns": [
            "Do not interpret wirelength-only movement from higher PHR budget as a legality fix.",
        ],
    },
    "memory_sensitivity": {
        "label": "Memory Sensitivity",
        "what_it_means": "Seed-locked memory variants materially change sampled actions or later-step behavior, so recurrent-state handling is influencing rollout quality.",
        "what_to_change_next": [
            "Inspect action deltas and memory traces starting at the first divergence step.",
            "Compare `carry`, `zero_each_step`, and `freeze_initial` on the same failing cases before modifying architecture.",
            "Treat memory reset or stabilization as an experimental control, not as an assumed improvement.",
        ],
        "experiments": [
            "Read `all_action_deltas.csv` and find the first step where action identity or controls diverge.",
            "Compare baseline and memory-variant traces on the same case around that first divergence step.",
            "Check whether memory changes best-so-far quality, final quality, or only wirelength.",
        ],
        "success_criteria": [
            "Memory variants no longer cause material overlap or pair-count divergence on failing cases.",
            "Action deltas remain small after early steps when memory is carried normally.",
        ],
        "anti_patterns": [
            "Do not treat memory as a transition-side factor under frozen actions; it only matters through action generation.",
        ],
    },
    "wire_recovery_missing": {
        "label": "Wire Recovery Missing",
        "what_it_means": "Legality is already strong, but the current refinement portfolio is not shortening wires without harming legality.",
        "what_to_change_next": [
            "Improve local post-legal refinement variants before touching DISCOVER again.",
            "Compare accepted local variants against incumbent_hold on the same saved basins.",
            "Tune refine-window construction and variant-specific move caps.",
        ],
        "experiments": [
            "Trace per-variant repaired candidates from the saved incumbent on the same fixed cases.",
            "Increase local reassignment and projection coverage without relaxing lexicographic legality checks.",
        ],
        "success_criteria": [
            "A nontrivial refine variant beats incumbent_hold on low-overlap validation cases.",
            "Diagnosis shifts away from memory as the main practical frontier.",
        ],
        "anti_patterns": [
            "Do not reopen DISCOVER memory work when legality has already saturated.",
        ],
    },
    "local_variant_sensitivity": {
        "label": "Local Variant Sensitivity",
        "what_it_means": "One local refinement variant dominates or one variant consistently breaks legality, so the cleanup portfolio is still under-tuned.",
        "what_to_change_next": [
            "Tune variant-specific locality and repair behavior.",
            "Gate or prune variants that consistently fail legality.",
        ],
        "experiments": [
            "Compare per-variant acceptance and legality rates on the fixed suite.",
            "Inspect losing variants' windows and repaired outcomes side by side with the winner.",
        ],
        "success_criteria": [
            "Variant wins are less brittle and clearly harmful variants stop breaking legality.",
        ],
        "anti_patterns": [
            "Do not assume one variant should dominate every case.",
        ],
    },
    "macro_refine_needed": {
        "label": "Macro Refine Needed",
        "what_it_means": "Wirelength pressure remains macro-driven, so standard-cell-only local cleanup is leaving recoverable wirelength on the table.",
        "what_to_change_next": [
            "Add macro-adjacent local refinement when macro-connected gradients dominate.",
            "Keep macro refinement bounded and repaired instead of globally reopening the layout.",
        ],
        "experiments": [
            "Measure whether high-gradient refine windows are macro-adjacent on the failing cases.",
            "Run a macro-inclusive local cleanup ablation on the same incumbents.",
        ],
        "success_criteria": [
            "Macro-adjacent cleanup improves wirelength without losing the current legality basin.",
        ],
        "anti_patterns": [
            "Do not convert REFINE into a global macro search stage.",
        ],
    },
}


def parse_csv_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def default_suite_seed_from_checkpoint(checkpoint):
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    train_seed = int(config.get("seed", 1234))
    return train_seed + 1_000_000


def default_sizes_from_checkpoint(checkpoint):
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    return str(config.get("sizes", DEFAULT_SIZES))


def default_validation_episodes_from_checkpoint(checkpoint):
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    return int(config.get("validation_episodes", 4))


def default_temperature_from_checkpoint(checkpoint, env_config):
    stats = checkpoint.get("stats", {}) if isinstance(checkpoint, dict) else {}
    if "soft_tau" in stats and stats["soft_tau"] is not None:
        return float(stats["soft_tau"])
    return float(env_config.soft_tau)


def maybe_run_validity_tests(repo_root: Path, output_dir: Path):
    command = [sys.executable, str(repo_root / "validity_tests.py")]
    result = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    log_path = output_dir / "validity_tests.txt"
    log_path.write_text(
        "COMMAND:\n"
        + " ".join(command)
        + "\n\nSTDOUT:\n"
        + result.stdout
        + "\nSTDERR:\n"
        + result.stderr,
        encoding="utf-8",
    )
    return {
        "command": command,
        "returncode": int(result.returncode),
        "passed": bool(result.returncode == 0),
        "log_path": str(log_path),
    }


def rank_validation_cases(
    *,
    policy,
    checkpoint,
    device,
    sizes_spec,
    validation_episodes,
    suite_seed,
    temperature,
    deterministic,
    relaxation,
):
    sizes = parse_sizes(sizes_spec)
    env_config = env_config_from_checkpoint(checkpoint, argparse.Namespace(
        steps=None,
        soft_relaxation=None,
        residual_flow=None,
        phr_layer=None,
        exact_audit=None,
        density=None,
        clusters=None,
        stop=None,
        fixed_pd_controls=None,
        ordering_representation=None,
        branch_mode=None,
        al_mode=None,
    ))
    validation_suite = build_validation_suite(sizes, validation_episodes, suite_seed)
    rows = []
    for suite_index, spec in enumerate(validation_suite):
        forced_size = tuple(spec["size"])
        episode_seed = int(spec["seed"])
        mode_infos = []
        for discover_mode in DISCOVER_MODE_NAMES:
            _transitions, info = collect_episode(
                policy,
                sizes,
                env_config,
                device,
                episode_seed,
                temperature,
                soft_tau=env_config.soft_tau,
                relaxation=relaxation,
                forced_size=forced_size,
                deterministic=deterministic,
                discover_mode=discover_mode,
                post_legal_refine_portfolio=True,
            )
            mode_infos.append(info)
        info = min(
            mode_infos,
            key=lambda row: (
                float(row["best_overlap"]),
                int(row["best_exact_overlap_pairs"]),
                float(row["best_wl"]),
            ),
        )
        mode_sensitivity = max(float(row["best_overlap"]) for row in mode_infos) - min(float(row["best_overlap"]) for row in mode_infos)
        rows.append(
            {
                "suite_index": int(suite_index),
                "size": f"{forced_size[0]}:{forced_size[1]}",
                "size_macros": int(forced_size[0]),
                "size_std_cells": int(forced_size[1]),
                "seed": int(episode_seed),
                "winning_discover_mode": str(info.get("discover_mode", "balanced")),
                "winning_refine_variant": str(info.get("winning_refine_variant", "incumbent_hold")),
                "mode_sensitivity": float(mode_sensitivity),
                "variant_sensitivity": float(
                    max(float(row.get("best_wl", 0.0)) for row in mode_infos)
                    - min(float(row.get("best_wl", 0.0)) for row in mode_infos)
                ),
                "best_overlap": float(info["best_overlap"]),
                "best_exact_overlap_pairs": int(info["best_exact_overlap_pairs"]),
                "best_wl": float(info["best_wl"]),
                "refine_window_size": int(info.get("refine_window_size", 0)),
                "branch_violation": float(info["branch_violation"]),
                "missed_pairs": int(info["missed_pairs"]),
                "audit_pressure_scale": float(info["audit_pressure_scale"]),
                "steps": int(info["steps"]),
                "phase_failure_score": float(info.get("phase_failure_score", 0.0)),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row.get("phase_failure_score", 0.0)),
            -float(row["best_overlap"]),
            -int(row["best_exact_overlap_pairs"]),
            -float(row["best_wl"]),
        )
    )
    return rows


def parse_numeric(value):
    if value == "" or value is None:
        return value
    try:
        if any(ch in str(value) for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_variant_summary(path: Path):
    rows = []
    for row in parse_csv_rows(path):
        rows.append({key: parse_numeric(value) for key, value in row.items()})
    return rows


def row_by_variant(rows, name):
    for row in rows:
        if row.get("variant_name") == name:
            return row
    return None


def _has_numeric_variant_summary(row):
    if not row:
        return False
    required = (
        "final_overlap_ratio",
        "best_overlap_ratio",
        "final_num_overlap_pairs",
        "best_num_overlap_pairs",
    )
    for key in required:
        value = row.get(key, "")
        if value is None or str(value).strip() == "":
            return False
    return True


def diagnose_case(case_summary_rows):
    baseline = row_by_variant(case_summary_rows, "baseline_action_locked")
    if baseline is None or not _has_numeric_variant_summary(baseline):
        return {
            "dominant_factor": "unresolved",
            "reason": "baseline counterfactual row missing",
        }

    horizon_rows = [row for row in case_summary_rows if row.get("variant_family") == "horizon_cutoff" and _has_numeric_variant_summary(row)]
    active_row = row_by_variant(case_summary_rows, "active_set_all_pairs")
    if not _has_numeric_variant_summary(active_row):
        active_row = None
    phr_rows = [row for row in case_summary_rows if row.get("variant_family") == "phr_budget" and _has_numeric_variant_summary(row)]
    memory_rows = [
        row
        for row in case_summary_rows
        if row.get("variant_family") == "memory"
        and row.get("memory_mode") != "carry"
        and _has_numeric_variant_summary(row)
    ]

    baseline_final_overlap = float(baseline["final_overlap_ratio"])
    baseline_best_overlap = float(baseline["best_overlap_ratio"])
    baseline_final_pairs = int(baseline["final_num_overlap_pairs"])
    baseline_best_pairs = int(baseline["best_num_overlap_pairs"])

    horizon_best = None
    if horizon_rows:
        horizon_best = min(
            horizon_rows,
            key=lambda row: (
                float(row["final_overlap_ratio"]),
                int(row["final_num_overlap_pairs"]),
                float(row["final_normalized_wl"]),
            ),
        )
    phr_best = None
    if phr_rows:
        phr_best = min(
            phr_rows,
            key=lambda row: (
                float(row["final_overlap_ratio"]),
                int(row["final_num_overlap_pairs"]),
                float(row["final_normalized_wl"]),
            ),
        )

    horizon_overlap_gain = 0.0 if horizon_best is None else baseline_final_overlap - float(horizon_best["final_overlap_ratio"])
    horizon_pair_gain = 0 if horizon_best is None else baseline_final_pairs - int(horizon_best["final_num_overlap_pairs"])

    active_overlap_gain = 0.0
    active_pair_gain = 0
    if active_row is not None:
        active_overlap_gain = max(
            baseline_final_overlap - float(active_row["final_overlap_ratio"]),
            baseline_best_overlap - float(active_row["best_overlap_ratio"]),
        )
        active_pair_gain = max(
            baseline_final_pairs - int(active_row["final_num_overlap_pairs"]),
            baseline_best_pairs - int(active_row["best_num_overlap_pairs"]),
        )

    phr_overlap_gain = 0.0
    phr_pair_gain = 0
    if phr_best is not None:
        phr_overlap_gain = max(
            baseline_final_overlap - float(phr_best["final_overlap_ratio"]),
            baseline_best_overlap - float(phr_best["best_overlap_ratio"]),
        )
        phr_pair_gain = max(
            baseline_final_pairs - int(phr_best["final_num_overlap_pairs"]),
            baseline_best_pairs - int(phr_best["best_num_overlap_pairs"]),
        )

    memory_overlap_sensitivity = 0.0
    memory_pair_sensitivity = 0
    if memory_rows:
        memory_overlap_sensitivity = max(
            max(abs(float(row["delta_vs_baseline_final_overlap_ratio"])) for row in memory_rows),
            max(abs(float(row["delta_vs_baseline_best_overlap_ratio"])) for row in memory_rows),
        )
        memory_pair_sensitivity = max(
            max(abs(int(row["delta_vs_baseline_final_num_overlap_pairs"])) for row in memory_rows),
            max(abs(int(row["delta_vs_baseline_best_num_overlap_pairs"])) for row in memory_rows),
        )

    continuation_score = max(horizon_overlap_gain, 0.0) + 0.02 * max(horizon_pair_gain, 0)
    active_set_score = max(active_overlap_gain, 0.0) + 0.02 * max(active_pair_gain, 0)
    phr_score = max(phr_overlap_gain, 0.0) + 0.02 * max(phr_pair_gain, 0)
    memory_score = max(memory_overlap_sensitivity, 0.0) + 0.02 * max(memory_pair_sensitivity, 0)

    scores = {
        "continuation_after_good_basin": continuation_score,
        "active_set_incompleteness": active_set_score,
        "phr_budget_sensitivity": phr_score,
        "memory_sensitivity": memory_score,
    }
    dominant_factor = max(scores, key=scores.get)

    reason_lines = []
    if horizon_best is not None:
        reason_lines.append(
            f"best horizon cutoff is {int(horizon_best['horizon_cutoff'])} with final overlap {float(horizon_best['final_overlap_ratio']):.6f} versus baseline {baseline_final_overlap:.6f}"
        )
    if active_row is not None:
        reason_lines.append(
            f"all-pairs active-set delta is {baseline_final_overlap - float(active_row['final_overlap_ratio']):.6f} overlap and {baseline_final_pairs - int(active_row['final_num_overlap_pairs'])} pairs"
        )
    if phr_best is not None:
        reason_lines.append(
            f"best PHR-budget variant is pd_steps={phr_best.get('phr_pd_steps_override')} with final overlap {float(phr_best['final_overlap_ratio']):.6f}"
        )
    if memory_rows:
        reason_lines.append(
            f"max memory sensitivity is {memory_overlap_sensitivity:.6f} overlap and {memory_pair_sensitivity} pairs"
        )

    recommendation = {
        "continuation_after_good_basin": "Focus on stop / horizon / best-so-far preservation. The rollout finds a good basin early and degrades afterward.",
        "active_set_incompleteness": "Focus on exact-audit coverage and constraint exposure. The transition improves when all pairs are visible.",
        "phr_budget_sensitivity": "Focus on primal-dual budget and inner optimization controls. Legality depends materially on PHR step budget.",
        "memory_sensitivity": "Focus on recurrent-state handling. Hidden-state drift is changing action quality materially after early steps.",
    }[dominant_factor]
    playbook = FACTOR_PLAYBOOK[dominant_factor]

    return {
        "dominant_factor": dominant_factor,
        "continuation_score": continuation_score,
        "active_set_score": active_set_score,
        "phr_score": phr_score,
        "memory_score": memory_score,
        "recommendation": recommendation,
        "reason": " | ".join(reason_lines),
        "playbook_label": playbook["label"],
        "what_it_means": playbook["what_it_means"],
        "what_to_change_next": list(playbook["what_to_change_next"]),
        "experiments": list(playbook["experiments"]),
        "success_criteria": list(playbook["success_criteria"]),
        "anti_patterns": list(playbook["anti_patterns"]),
    }


def run_counterfactual_case(
    *,
    repo_root,
    checkpoint,
    device,
    case_row,
    steps,
    temperature,
    phr_step_variants,
    horizon_cutoffs,
    all_pairs_threshold,
    full_pair_matrix_threshold,
    output_dir,
):
    size = str(case_row["size"])
    seed = int(case_row["seed"])
    command = [
        sys.executable,
        str(repo_root / "counterfactual_replay.py"),
        "--checkpoint",
        str(checkpoint),
        "--device",
        str(device),
        "--size",
        str(size),
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--temperature",
        str(temperature),
        "--deterministic",
        "--phr-step-variants",
        str(phr_step_variants),
        "--horizon-cutoffs",
        str(horizon_cutoffs),
        "--all-pairs-threshold",
        str(all_pairs_threshold),
        "--full-pair-matrix-threshold",
        str(full_pair_matrix_threshold),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=str(repo_root), check=True)
    return command


def render_report(
    *,
    path,
    checkpoint,
    validity_result,
    validation_rows,
    analyzed_rows,
    wisdom_result,
):
    lines = [
        "# Diagnosis Suite",
        "",
        f"- checkpoint: `{checkpoint}`",
        "",
    ]
    if validity_result is not None:
        lines.extend(
            [
                "## Validity Tests",
                "",
                f"- passed: `{validity_result['passed']}`",
                f"- log: `{validity_result['log_path']}`",
                "",
            ]
        )

    lines.extend(["## Validation Ranking", ""])
    for row in validation_rows:
        lines.append(
            "- suite `{idx}` size `{size}` seed `{seed}` mode `{mode}`: overlap `{ov:.6f}`, pairs `{pairs}`, wl `{wl:.6f}`, mode-sensitivity `{sens:.6f}`".format(
                idx=row["suite_index"],
                size=row["size"],
                seed=row["seed"],
                mode=row.get("winning_discover_mode", "balanced"),
                ov=row["best_overlap"],
                pairs=row["best_exact_overlap_pairs"],
                wl=row["best_wl"],
                sens=float(row.get("mode_sensitivity", 0.0)),
            )
        )
    lines.append("")

    lines.extend(["## Analyzed Cases", ""])
    for row in analyzed_rows:
        lines.append(
            "- suite `{idx}` size `{size}` seed `{seed}`: dominant factor `{factor}`".format(
                idx=row["suite_index"],
                size=row["size"],
                seed=row["seed"],
                factor=row["dominant_factor"],
            )
        )
        lines.append(f"  reason: {row['reason']}")
        lines.append(f"  recommendation: {row['recommendation']}")
        lines.append(f"  meaning: {row['what_it_means']}")
        lines.append("  next experiments:")
        for item in row["experiments"]:
            lines.append(f"    - {item}")
        lines.append("  success criteria:")
        for item in row["success_criteria"]:
            lines.append(f"    - {item}")
        lines.append("  anti-patterns:")
        for item in row["anti_patterns"]:
            lines.append(f"    - {item}")
    lines.append("")

    if wisdom_result is not None:
        lines.extend(["## Wisdom Gaps", ""])
        summary = wisdom_result["summary"]
        primary_counts = summary.get("primary_gap_counts", {})
        for label, count in sorted(primary_counts.items(), key=lambda item: (-int(item[1]), item[0])):
            lines.append(f"- `{label}`: `{count}` cases")
        lines.append("")
        lines.extend(["## Wisdom Roadmap", ""])
        for row in wisdom_result.get("roadmap", [])[:5]:
            lines.append(
                "- rank `{rank}` wisdom `{wisdom}` via `{mechanism}` ({bucket})".format(
                    rank=row["priority_rank"],
                    wisdom=row["wisdom_class"],
                    mechanism=row["transfer_mechanism"],
                    bucket=row["transfer_bucket"],
                )
            )
            lines.append(f"  transfer path: {row['transfer_path']}")
            lines.append(f"  success metric: {row['success_metric']}")
        lines.append("")

    lines.extend(
        [
            "## Breadcrumbs",
            "",
            "- Always preserve the generated `validation_case_ranking.csv`, `case_diagnosis.csv`, and `diagnosis_summary.json` together.",
            "- If a future run disagrees with the current diagnosis, compare the per-case counterfactual subdirectories rather than only aggregate training logs.",
            "- Do not interpret memory as a transition-side cause under fixed actions; use the resampled memory variants for that question.",
            "- Do not blame active-set incompleteness unless the `active_set_all_pairs` variant materially improves legality.",
            "- Do not blame PHR budget unless the `phr_pd_steps_*` ladder materially changes legality on the same case.",
            "- If cutoff variants preserve legality while the full horizon degrades, treat continuation as the primary failure mode first.",
            "",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_case_next_steps_entry(row):
    return {
        "suite_index": int(row["suite_index"]),
        "size": str(row["size"]),
        "seed": int(row["seed"]),
        "dominant_factor": str(row["dominant_factor"]),
        "playbook_label": str(row["playbook_label"]),
        "recommendation": str(row["recommendation"]),
        "what_it_means": str(row["what_it_means"]),
        "what_to_change_next": list(row["what_to_change_next"]),
        "experiments": list(row["experiments"]),
        "success_criteria": list(row["success_criteria"]),
        "anti_patterns": list(row["anti_patterns"]),
        "counterfactual_dir": str(row["counterfactual_dir"]),
        "counterfactual_command": str(row["counterfactual_command"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sizes", default=None)
    parser.add_argument("--validation-episodes", type=int, default=None)
    parser.add_argument("--validation-suite-seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--relaxation", default="sigmoid", choices=["sigmoid", "neuralsort", "gumbel_sinkhorn"])
    parser.add_argument("--top-k-cases", type=int, default=2)
    parser.add_argument("--phr-step-variants", default="1,2,4,8,12,16")
    parser.add_argument("--horizon-cutoffs", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--all-pairs-threshold", type=int, default=64)
    parser.add_argument("--full-pair-matrix-threshold", type=int, default=40)
    parser.add_argument("--run-validity-tests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    policy, checkpoint = load_policy_checkpoint(args.checkpoint, device)
    policy.eval()
    env_config = env_config_from_checkpoint(
        checkpoint,
        argparse.Namespace(
            steps=args.steps,
            soft_relaxation=None,
            residual_flow=None,
            phr_layer=None,
            exact_audit=None,
            density=None,
            clusters=None,
            stop=None,
            fixed_pd_controls=None,
            ordering_representation=None,
            branch_mode=None,
            al_mode=None,
        ),
    )

    sizes_spec = args.sizes or default_sizes_from_checkpoint(checkpoint)
    validation_episodes = int(args.validation_episodes or default_validation_episodes_from_checkpoint(checkpoint))
    validation_suite_seed = int(args.validation_suite_seed or default_suite_seed_from_checkpoint(checkpoint))
    temperature = float(args.temperature if args.temperature is not None else default_temperature_from_checkpoint(checkpoint, env_config))
    steps = int(args.steps or env_config.horizon)

    validity_result = None
    if args.run_validity_tests:
        validity_result = maybe_run_validity_tests(repo_root, output_dir)

    validation_rows = rank_validation_cases(
        policy=policy,
        checkpoint=checkpoint,
        device=device,
        sizes_spec=sizes_spec,
        validation_episodes=validation_episodes,
        suite_seed=validation_suite_seed,
        temperature=temperature,
        deterministic=True,
        relaxation=args.relaxation,
    )
    write_csv(
        output_dir / "validation_case_ranking.csv",
        validation_rows,
        list(validation_rows[0].keys()) if validation_rows else [
            "suite_index",
            "size",
            "size_macros",
            "size_std_cells",
            "seed",
            "best_overlap",
            "best_exact_overlap_pairs",
            "best_wl",
            "branch_violation",
            "missed_pairs",
            "audit_pressure_scale",
            "steps",
        ],
    )

    analyzed_rows = []
    next_steps_entries = []
    for case_rank, case_row in enumerate(validation_rows[: max(int(args.top_k_cases), 0)]):
        case_name = "suite_{idx}_{size}_seed{seed}".format(
            idx=int(case_row["suite_index"]),
            size=str(case_row["size"]).replace(":", "m_") + "s",
            seed=int(case_row["seed"]),
        )
        case_dir = output_dir / case_name
        command = run_counterfactual_case(
            repo_root=repo_root,
            checkpoint=args.checkpoint,
            device=args.device,
            case_row=case_row,
            steps=steps,
            temperature=temperature,
            phr_step_variants=args.phr_step_variants,
            horizon_cutoffs=args.horizon_cutoffs,
            all_pairs_threshold=args.all_pairs_threshold,
            full_pair_matrix_threshold=args.full_pair_matrix_threshold,
            output_dir=case_dir,
        )
        variant_rows = load_variant_summary(case_dir / "variant_summary.csv")
        diagnosis = diagnose_case(variant_rows)
        if float(case_row.get("best_overlap", 1.0)) <= 0.15 and int(case_row.get("best_exact_overlap_pairs", 9999)) <= 1:
            if str(case_row.get("winning_refine_variant", "incumbent_hold")) == "incumbent_hold":
                playbook = FACTOR_PLAYBOOK["wire_recovery_missing"]
                diagnosis.update(
                    {
                        "dominant_factor": "wire_recovery_missing",
                        "recommendation": "Focus on legal wirelength recovery. The basin is already strong, but incumbent_hold is still winning the local cleanup portfolio.",
                        "playbook_label": playbook["label"],
                        "what_it_means": playbook["what_it_means"],
                        "what_to_change_next": list(playbook["what_to_change_next"]),
                        "experiments": list(playbook["experiments"]),
                        "success_criteria": list(playbook["success_criteria"]),
                        "anti_patterns": list(playbook["anti_patterns"]),
                    }
                )
            elif float(case_row.get("variant_sensitivity", 0.0)) > 0.10:
                playbook = FACTOR_PLAYBOOK["local_variant_sensitivity"]
                diagnosis.update(
                    {
                        "dominant_factor": "local_variant_sensitivity",
                        "recommendation": "Focus on tuning the local refinement portfolio. Variants differ materially on the same already-strong legality basin.",
                        "playbook_label": playbook["label"],
                        "what_it_means": playbook["what_it_means"],
                        "what_to_change_next": list(playbook["what_to_change_next"]),
                        "experiments": list(playbook["experiments"]),
                        "success_criteria": list(playbook["success_criteria"]),
                        "anti_patterns": list(playbook["anti_patterns"]),
                    }
                )
        analyzed_rows.append(
            {
                "case_rank": int(case_rank),
                "suite_index": int(case_row["suite_index"]),
                "size": str(case_row["size"]),
                "seed": int(case_row["seed"]),
                "winning_discover_mode": str(case_row.get("winning_discover_mode", "balanced")),
                "winning_refine_variant": str(case_row.get("winning_refine_variant", "incumbent_hold")),
                "variant_sensitivity": float(case_row.get("variant_sensitivity", 0.0)),
                "best_overlap": float(case_row["best_overlap"]),
                "best_exact_overlap_pairs": int(case_row["best_exact_overlap_pairs"]),
                "best_wl": float(case_row["best_wl"]),
                "counterfactual_dir": str(case_dir),
                "counterfactual_command": " ".join(command),
                **diagnosis,
            }
        )
        case_next_steps = build_case_next_steps_entry(analyzed_rows[-1])
        (case_dir / "case_next_steps.json").write_text(
            json.dumps(case_next_steps, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        next_steps_entries.append(case_next_steps)

    wisdom_result = write_wisdom_outputs(
        repo_root=repo_root,
        diagnosis_dir=output_dir,
        validation_rows=validation_rows,
        analyzed_rows=analyzed_rows,
    )
    structural_result = write_structural_outputs(
        repo_root=repo_root,
        diagnosis_dir=output_dir,
        validation_rows=validation_rows,
        analyzed_rows=analyzed_rows,
    )

    if analyzed_rows:
        write_csv(output_dir / "case_diagnosis.csv", analyzed_rows, list(analyzed_rows[0].keys()))
        (output_dir / "next_steps.json").write_text(
            json.dumps({"cases": next_steps_entries}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    summary_payload = {
        "checkpoint": str(args.checkpoint),
        "sizes": str(sizes_spec),
        "validation_episodes": int(validation_episodes),
        "validation_suite_seed": int(validation_suite_seed),
        "temperature": float(temperature),
        "steps": int(steps),
        "top_k_cases": int(args.top_k_cases),
        "validity_result": validity_result,
        "validation_rows": validation_rows,
        "analyzed_rows": analyzed_rows,
        "output_files": {
            "validation_case_ranking_csv": str(output_dir / "validation_case_ranking.csv"),
        },
    }
    if analyzed_rows:
        summary_payload["output_files"]["case_diagnosis_csv"] = str(output_dir / "case_diagnosis.csv")
        summary_payload["output_files"]["next_steps_json"] = str(output_dir / "next_steps.json")
    summary_payload["wisdom_audit"] = {
        "summary": wisdom_result["summary"],
        "roadmap": wisdom_result["roadmap"],
    }
    summary_payload["structural_audit"] = {
        "summary": structural_result["report"]["summary"],
        "next_architecture_experiment": structural_result["report"]["next_architecture_experiment"],
    }
    summary_payload["output_files"].update(wisdom_result["output_files"])
    summary_payload["output_files"].update(structural_result["output_files"])
    if validity_result is not None:
        summary_payload["output_files"]["validity_tests_log"] = str(output_dir / "validity_tests.txt")
    summary_payload["playbook"] = FACTOR_PLAYBOOK

    (output_dir / "diagnosis_summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_report(
        path=output_dir / "README.md",
        checkpoint=args.checkpoint,
        validity_result=validity_result,
        validation_rows=validation_rows,
        analyzed_rows=analyzed_rows,
        wisdom_result=wisdom_result,
    )


if __name__ == "__main__":
    main()
