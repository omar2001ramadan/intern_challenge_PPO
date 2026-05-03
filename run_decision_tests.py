"""Evaluate whether the current diagnosis interpretation is supported.

This is a thin analysis layer on top of a completed diagnosis suite directory.

Inputs:
- case_diagnosis.csv
- per-case variant_summary.csv
- per-case all_variant_steps.csv

Outputs:
- decision_test_results.json
- decision_test_report.md

The goal is to make the current interpretation explicit:
1. continuation / best-so-far preservation is first-order
2. memory sensitivity is second-order but real
3. active-set and PHR budget are not first-order unless falsified
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


OVERLAP_TOL = 0.03


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value, default=None):
    try:
        text = "" if value is None else str(value).strip()
        if text == "":
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _numeric_variant_rows(rows):
    numeric = []
    for row in rows:
        overlap = _safe_float(row.get("final_overlap_ratio"))
        if overlap is None:
            continue
        numeric.append(row)
    return numeric


def group_rows(rows, key):
    grouped = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def case_dir_from_row(row):
    return Path(row["counterfactual_dir"])


def summarize_case(case_row):
    case_dir = case_dir_from_row(case_row)
    raw_variant_rows = read_csv(case_dir / "variant_summary.csv")
    variant_rows = _numeric_variant_rows(raw_variant_rows)
    step_rows = read_csv(case_dir / "all_variant_steps.csv")

    unresolved = {
        "suite_index": int(case_row["suite_index"]),
        "size": case_row["size"],
        "seed": int(case_row["seed"]),
        "dominant_factor": case_row.get("dominant_factor", ""),
        "baseline_final_overlap": None,
        "baseline_best_overlap": None,
        "best_horizon_variant": "",
        "best_horizon_final_overlap": None,
        "continuation_improvement": 0.0,
        "all_pairs_final_overlap": None,
        "active_set_improvement": 0.0,
        "best_phr_variant": "",
        "best_phr_final_overlap": None,
        "phr_improvement": 0.0,
        "best_memory_variant": "",
        "best_memory_final_overlap": None,
        "memory_best_improvement": 0.0,
        "worst_memory_variant": "",
        "worst_memory_final_overlap": None,
        "memory_worst_regression": 0.0,
        "memory_abs_effect": 0.0,
        "reward_on_regression_steps": [],
        "has_positive_reward_on_regression": False,
        "resolved": False,
        "skip_reason": "missing_variant_summary",
    }

    if not variant_rows:
        return unresolved

    baseline = next((row for row in variant_rows if row["variant_name"] == "baseline_action_locked"), None)
    if baseline is None:
        unresolved["skip_reason"] = "missing_numeric_baseline"
        return unresolved
    baseline_final_overlap = float(baseline["final_overlap_ratio"])
    baseline_best_overlap = float(baseline["best_overlap_ratio"])

    horizon_rows = [row for row in variant_rows if row["variant_family"] == "horizon_cutoff"]
    all_pairs_row = next((row for row in variant_rows if row["variant_name"] == "active_set_all_pairs"), None)
    phr_rows = [row for row in variant_rows if row["variant_family"] == "phr_budget"]
    memory_rows = [row for row in variant_rows if row["variant_family"] == "memory" and row["variant_name"] != "memory_carry"]

    if not horizon_rows or all_pairs_row is None or not phr_rows or not memory_rows:
        unresolved.update(
            {
                "baseline_final_overlap": baseline_final_overlap,
                "baseline_best_overlap": baseline_best_overlap,
                "skip_reason": "missing_counterfactual_family",
            }
        )
        return unresolved

    best_horizon = min(horizon_rows, key=lambda row: float(row["final_overlap_ratio"]))
    best_phr = min(phr_rows, key=lambda row: float(row["final_overlap_ratio"]))

    best_memory_help = min(memory_rows, key=lambda row: float(row["final_overlap_ratio"]))
    worst_memory_help = max(memory_rows, key=lambda row: float(row["final_overlap_ratio"]))

    continuation_improvement = baseline_final_overlap - float(best_horizon["final_overlap_ratio"])
    active_set_improvement = baseline_final_overlap - float(all_pairs_row["final_overlap_ratio"])
    phr_improvement = baseline_final_overlap - float(best_phr["final_overlap_ratio"])
    memory_best_improvement = baseline_final_overlap - float(best_memory_help["final_overlap_ratio"])
    memory_worst_regression = float(worst_memory_help["final_overlap_ratio"]) - baseline_final_overlap
    memory_abs_effect = max(abs(memory_best_improvement), abs(memory_worst_regression))

    baseline_steps = sorted(
        [row for row in step_rows if row["variant_name"] == "baseline_action_locked"],
        key=lambda row: int(row["step"]),
    )
    reward_on_regression_steps = []
    running_best = None
    for row in baseline_steps:
        post_overlap = _safe_float(row.get("post_overlap_ratio"))
        reward = _safe_float(row.get("reward"))
        if post_overlap is None or reward is None:
            continue
        if running_best is None:
            running_best = post_overlap
            continue
        if post_overlap > running_best + 1e-9 and reward > 0.0:
            reward_on_regression_steps.append(
                {
                    "step": int(row["step"]),
                    "post_overlap": post_overlap,
                    "running_best_before_step": running_best,
                    "reward": reward,
                }
            )
        running_best = min(running_best, post_overlap)

    return {
        "suite_index": int(case_row["suite_index"]),
        "size": case_row["size"],
        "seed": int(case_row["seed"]),
        "dominant_factor": case_row["dominant_factor"],
        "baseline_final_overlap": baseline_final_overlap,
        "baseline_best_overlap": baseline_best_overlap,
        "best_horizon_variant": best_horizon["variant_name"],
        "best_horizon_final_overlap": float(best_horizon["final_overlap_ratio"]),
        "continuation_improvement": continuation_improvement,
        "all_pairs_final_overlap": float(all_pairs_row["final_overlap_ratio"]),
        "active_set_improvement": active_set_improvement,
        "best_phr_variant": best_phr["variant_name"],
        "best_phr_final_overlap": float(best_phr["final_overlap_ratio"]),
        "phr_improvement": phr_improvement,
        "best_memory_variant": best_memory_help["variant_name"],
        "best_memory_final_overlap": float(best_memory_help["final_overlap_ratio"]),
        "memory_best_improvement": memory_best_improvement,
        "worst_memory_variant": worst_memory_help["variant_name"],
        "worst_memory_final_overlap": float(worst_memory_help["final_overlap_ratio"]),
        "memory_worst_regression": memory_worst_regression,
        "memory_abs_effect": memory_abs_effect,
        "reward_on_regression_steps": reward_on_regression_steps,
        "has_positive_reward_on_regression": bool(reward_on_regression_steps),
        "resolved": True,
        "skip_reason": "",
    }


def build_results(case_summaries):
    resolved_cases = [case for case in case_summaries if case.get("resolved", True)]
    unresolved_cases = [case for case in case_summaries if not case.get("resolved", True)]
    continuation_supported_cases = [
        case for case in resolved_cases
        if case["continuation_improvement"] >= OVERLAP_TOL
    ]
    active_set_supported_cases = [
        case for case in resolved_cases
        if case["active_set_improvement"] >= OVERLAP_TOL
    ]
    phr_supported_cases = [
        case for case in resolved_cases
        if case["phr_improvement"] >= OVERLAP_TOL
    ]
    memory_supported_cases = [
        case for case in resolved_cases
        if case["memory_abs_effect"] >= OVERLAP_TOL
    ]
    reward_regression_cases = [
        case for case in resolved_cases
        if case["has_positive_reward_on_regression"]
    ]
    continuation_unsupported_cases = [
        case for case in resolved_cases
        if case["continuation_improvement"] < OVERLAP_TOL
    ]
    memory_unsupported_cases = [
        case for case in resolved_cases
        if case["memory_abs_effect"] < OVERLAP_TOL
    ]

    results = {
        "thresholds": {
            "overlap_materiality": OVERLAP_TOL,
            "multi_case_falsifier_count": 2,
            "total_cases": len(case_summaries),
            "resolved_cases": len(resolved_cases),
            "unresolved_cases": len(unresolved_cases),
        },
        "hypotheses": {
            "continuation_primary": {
                "supported": len(continuation_supported_cases) >= 2,
                "support_count": len(continuation_supported_cases),
                "cases": [f"{case['size']} seed {case['seed']}" for case in continuation_supported_cases],
            },
            "memory_secondary_but_real": {
                "supported": len(memory_supported_cases) >= 2,
                "support_count": len(memory_supported_cases),
                "cases": [f"{case['size']} seed {case['seed']}" for case in memory_supported_cases],
            },
            "reward_pays_for_late_regression": {
                "supported": len(reward_regression_cases) >= 2,
                "support_count": len(reward_regression_cases),
                "cases": [f"{case['size']} seed {case['seed']}" for case in reward_regression_cases],
            },
        },
        "falsifiers": {
            "active_set_primary": {
                "triggered": len(active_set_supported_cases) >= 2,
                "support_count": len(active_set_supported_cases),
                "cases": [f"{case['size']} seed {case['seed']}" for case in active_set_supported_cases],
            },
            "phr_primary": {
                "triggered": len(phr_supported_cases) >= 2,
                "support_count": len(phr_supported_cases),
                "cases": [f"{case['size']} seed {case['seed']}" for case in phr_supported_cases],
            },
            "continuation_not_primary": {
                "triggered": len(continuation_supported_cases) < 2,
                "count": len(continuation_unsupported_cases),
                "cases": [f"{case['size']} seed {case['seed']}" for case in continuation_unsupported_cases],
            },
            "memory_not_material": {
                "triggered": len(memory_supported_cases) < 2,
                "count": len(memory_unsupported_cases),
                "cases": [f"{case['size']} seed {case['seed']}" for case in memory_unsupported_cases],
            },
        },
        "cases": case_summaries,
        "unresolved_cases": [
            {
                "label": f"{case['size']} seed {case['seed']}",
                "skip_reason": case.get("skip_reason", ""),
            }
            for case in unresolved_cases
        ],
    }
    return results


def render_report(results):
    lines = [
        "# Decision Test Report",
        "",
        f"- overlap materiality threshold: `{results['thresholds']['overlap_materiality']}`",
        "",
        "## Hypotheses",
    ]
    for name, payload in results["hypotheses"].items():
        lines.append(f"- `{name}`: supported = `{payload['supported']}`, count = `{payload['support_count']}`, cases = {payload['cases']}")
    lines.extend(["", "## Falsifiers"])
    for name, payload in results["falsifiers"].items():
        count = payload.get("count", payload.get("support_count"))
        lines.append(f"- `{name}`: triggered = `{payload['triggered']}`, count = `{count}`, cases = {payload['cases']}")
    lines.extend(["", "## Per Case"])
    for case in results["cases"]:
        if not case.get("resolved", True):
            lines.append(f"- `{case['size']} seed {case['seed']}`: unresolved (`{case.get('skip_reason', '')}`)")
            continue
        lines.append(
            f"- `{case['size']} seed {case['seed']}`: continuation improvement `{case['continuation_improvement']:.6f}`, "
            f"active-set improvement `{case['active_set_improvement']:.6f}`, "
            f"PHR improvement `{case['phr_improvement']:.6f}`, "
            f"memory abs effect `{case['memory_abs_effect']:.6f}`, "
            f"positive-reward regression `{case['has_positive_reward_on_regression']}`"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnosis-dir", required=True)
    args = parser.parse_args()

    diagnosis_dir = Path(args.diagnosis_dir).resolve()
    case_rows = read_csv(diagnosis_dir / "case_diagnosis.csv")
    case_summaries = [summarize_case(row) for row in case_rows]
    results = build_results(case_summaries)

    results_path = diagnosis_dir / "decision_test_results.json"
    report_path = diagnosis_dir / "decision_test_report.md"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(results), encoding="utf-8")

    print(f"Wrote {results_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
