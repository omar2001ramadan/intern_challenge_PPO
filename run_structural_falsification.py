"""Run structural falsification experiments against the learned candidate ranker.

Current focus:
- mask the external repaired-candidate selector inside each discover mode
- let the learned candidate ranker choose from the same repaired candidate set
- measure regret against the external lexicographic winner on the fixed suite
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from ordering_policy import load_policy_checkpoint
from run_diagnosis_suite import (
    DEFAULT_SIZES,
    default_sizes_from_checkpoint,
    default_suite_seed_from_checkpoint,
    default_temperature_from_checkpoint,
    default_validation_episodes_from_checkpoint,
)
from train_ppo import (
    build_validation_suite,
    default_device_arg,
    parse_sizes,
    validate_policy,
)
from visualize_rollout_trace import env_config_from_checkpoint


def _write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _metrics(record):
    return (
        float(record.get("candidate_overlap", 0.0)),
        int(record.get("candidate_pairs", 0)),
        float(record.get("candidate_wirelength", 0.0)),
    )


def _same_metrics(lhs, rhs):
    return _metrics(lhs) == _metrics(rhs)


def _graph_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _graph_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_graph_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_graph_to_device(item, device) for item in value)
    return value


def _arg_namespace_for_env(steps=None):
    return argparse.Namespace(
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
        steps=steps,
    )


def build_mode_falsification_rows(policy, validation_rows, *, device):
    rows = []
    for fallback_index, case_row in enumerate(validation_rows):
        suite_index = int(case_row.get("suite_index", fallback_index))
        case_seed = int(case_row.get("seed", fallback_index))
        case_size = str(case_row.get("size", ""))
        for mode_row in case_row.get("per_mode_info_rows", []):
            rows.append(
                {
                    "suite_index": suite_index,
                    "size": case_size,
                    "seed": case_seed,
                    "discover_mode": str(mode_row["discover_mode"]),
                    "candidate_count": int(sum(1 for item in mode_row.get("candidate_records", []) if bool(item.get("candidate_live_input", False)))),
                    "chooser_selected_source": str(mode_row.get("chooser_selected_source", "rollout_best")),
                    "external_source": str(mode_row.get("external_selected_source", "rollout_best")),
                    "chooser_selected_overlap": float(mode_row.get("chooser_selected_overlap", mode_row.get("best_overlap", 0.0))),
                    "chooser_selected_pairs": int(mode_row.get("chooser_selected_pairs", mode_row.get("best_exact_overlap_pairs", 0))),
                    "chooser_selected_wirelength": float(mode_row.get("chooser_selected_wirelength", mode_row.get("best_wl", 0.0))),
                    "external_overlap": float(mode_row.get("external_selected_overlap", mode_row.get("best_overlap", 0.0))),
                    "external_pairs": int(mode_row.get("external_selected_pairs", mode_row.get("best_exact_overlap_pairs", 0))),
                    "external_wirelength": float(mode_row.get("external_selected_wirelength", mode_row.get("best_wl", 0.0))),
                    "overlap_regret": float(mode_row.get("chooser_regret_overlap", 0.0)),
                    "pair_regret": int(mode_row.get("chooser_regret_pairs", 0)),
                    "wire_regret": float(mode_row.get("chooser_regret_wirelength", 0.0)),
                    "metric_match": bool(mode_row.get("chooser_match", True)),
                    "predicted_source": str(mode_row.get("chooser_selected_source", "rollout_best")),
                    "predicted_overlap": float(mode_row.get("chooser_selected_overlap", mode_row.get("best_overlap", 0.0))),
                    "predicted_pairs": int(mode_row.get("chooser_selected_pairs", mode_row.get("best_exact_overlap_pairs", 0))),
                    "predicted_wirelength": float(mode_row.get("chooser_selected_wirelength", mode_row.get("best_wl", 0.0))),
                    "predicted_score": 0.0,
                    "external_score": 0.0,
                }
            )
    return rows


def build_case_falsification_rows(validation_rows, mode_rows):
    by_case = {}
    for row in mode_rows:
        key = (int(row["suite_index"]), int(row["seed"]))
        by_case.setdefault(key, []).append(row)

    rows = []
    for fallback_index, case_row in enumerate(validation_rows):
        suite_index = int(case_row.get("suite_index", fallback_index))
        case_seed = int(case_row.get("seed", fallback_index))
        key = (suite_index, case_seed)
        per_mode = by_case.get(key, [])
        if not per_mode:
            continue

        external_case_winner = {
            "discover_mode": str(case_row.get("external_winning_mode", case_row.get("winning_discover_mode", ""))),
            "candidate_source": str(case_row.get("external_selected_source", "rollout_best")),
            "candidate_overlap": float(case_row.get("external_selected_overlap", case_row.get("best_overlap", 0.0))),
            "candidate_pairs": int(case_row.get("external_selected_pairs", case_row.get("best_exact_overlap_pairs", 0))),
            "candidate_wirelength": float(case_row.get("external_selected_wirelength", case_row.get("best_wl", 0.0))),
        }
        predicted_case_winner = min(
            [
                {
                    "discover_mode": str(row["discover_mode"]),
                    "candidate_overlap": float(row["chooser_selected_overlap"]),
                    "candidate_pairs": int(row["chooser_selected_pairs"]),
                    "candidate_wirelength": float(row["chooser_selected_wirelength"]),
                    "candidate_source": str(row["chooser_selected_source"]),
                }
                for row in per_mode
            ],
            key=lambda item: (float(item["candidate_overlap"]), int(item["candidate_pairs"]), float(item["candidate_wirelength"])),
        )
        rows.append(
            {
                "suite_index": suite_index,
                "size": str(case_row.get("size", "")),
                "seed": case_seed,
                "external_winning_mode": str(external_case_winner["discover_mode"]),
                "external_source": str(external_case_winner["candidate_source"]),
                "external_overlap": float(external_case_winner["candidate_overlap"]),
                "external_pairs": int(external_case_winner["candidate_pairs"]),
                "external_wirelength": float(external_case_winner["candidate_wirelength"]),
                "chooser_selected_winning_mode": str(predicted_case_winner["discover_mode"]),
                "chooser_selected_source": str(predicted_case_winner["candidate_source"]),
                "chooser_selected_overlap": float(predicted_case_winner["candidate_overlap"]),
                "chooser_selected_pairs": int(predicted_case_winner["candidate_pairs"]),
                "chooser_selected_wirelength": float(predicted_case_winner["candidate_wirelength"]),
                "overlap_regret": float(predicted_case_winner["candidate_overlap"]) - float(external_case_winner["candidate_overlap"]),
                "pair_regret": int(predicted_case_winner["candidate_pairs"]) - int(external_case_winner["candidate_pairs"]),
                "wire_regret": float(predicted_case_winner["candidate_wirelength"]) - float(external_case_winner["candidate_wirelength"]),
                "metric_match": bool(_same_metrics(predicted_case_winner, external_case_winner)),
                "predicted_winning_mode": str(predicted_case_winner["discover_mode"]),
                "predicted_source": str(predicted_case_winner["candidate_source"]),
                "predicted_overlap": float(predicted_case_winner["candidate_overlap"]),
                "predicted_pairs": int(predicted_case_winner["candidate_pairs"]),
                "predicted_wirelength": float(predicted_case_winner["candidate_wirelength"]),
            }
        )
    return rows


def build_summary(mode_rows, case_rows):
    mode_count = max(len(mode_rows), 1)
    case_count = max(len(case_rows), 1)
    return {
        "mode_level_rows": int(len(mode_rows)),
        "case_level_rows": int(len(case_rows)),
        "chooser_top1_match_rate_mode": sum(1.0 if row["metric_match"] else 0.0 for row in mode_rows) / mode_count,
        "chooser_top1_match_rate_case": sum(1.0 if row["metric_match"] else 0.0 for row in case_rows) / case_count,
        "chooser_mean_overlap_regret_mode": sum(float(row["overlap_regret"]) for row in mode_rows) / mode_count,
        "chooser_mean_pair_regret_mode": sum(float(row["pair_regret"]) for row in mode_rows) / mode_count,
        "chooser_mean_wire_regret_mode": sum(float(row["wire_regret"]) for row in mode_rows) / mode_count,
        "chooser_mean_overlap_regret_case": sum(float(row["overlap_regret"]) for row in case_rows) / case_count,
        "chooser_mean_pair_regret_case": sum(float(row["pair_regret"]) for row in case_rows) / case_count,
        "chooser_mean_wire_regret_case": sum(float(row["wire_regret"]) for row in case_rows) / case_count,
        "mode_match_rate": sum(1.0 if row["metric_match"] else 0.0 for row in mode_rows) / mode_count,
        "case_match_rate": sum(1.0 if row["metric_match"] else 0.0 for row in case_rows) / case_count,
        "mode_mean_overlap_regret": sum(float(row["overlap_regret"]) for row in mode_rows) / mode_count,
        "mode_mean_pair_regret": sum(float(row["pair_regret"]) for row in mode_rows) / mode_count,
        "mode_mean_wire_regret": sum(float(row["wire_regret"]) for row in mode_rows) / mode_count,
        "case_mean_overlap_regret": sum(float(row["overlap_regret"]) for row in case_rows) / case_count,
        "case_mean_pair_regret": sum(float(row["pair_regret"]) for row in case_rows) / case_count,
        "case_mean_wire_regret": sum(float(row["wire_regret"]) for row in case_rows) / case_count,
        "mode_overlap_worse_count": int(sum(1 for row in mode_rows if float(row["overlap_regret"]) > 0.0)),
        "mode_pair_worse_count": int(sum(1 for row in mode_rows if int(row["pair_regret"]) > 0)),
        "mode_wire_worse_count": int(sum(1 for row in mode_rows if float(row["wire_regret"]) > 0.0)),
        "case_overlap_worse_count": int(sum(1 for row in case_rows if float(row["overlap_regret"]) > 0.0)),
        "case_pair_worse_count": int(sum(1 for row in case_rows if int(row["pair_regret"]) > 0)),
        "case_wire_worse_count": int(sum(1 for row in case_rows if float(row["wire_regret"]) > 0.0)),
    }


def run_falsification(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    device: str,
    sizes_spec: str,
    validation_episodes: int,
    suite_seed: int,
    temperature: float,
):
    torch_device = torch.device(device)
    policy, checkpoint = load_policy_checkpoint(checkpoint_path, torch_device)
    env_config = env_config_from_checkpoint(checkpoint, _arg_namespace_for_env())
    sizes = parse_sizes(sizes_spec)
    validation_suite = build_validation_suite(sizes, validation_episodes, suite_seed)
    summary, validation_rows = validate_policy(
        policy,
        sizes,
        env_config,
        torch_device,
        suite_seed,
        temperature,
        soft_tau=float(env_config.soft_tau),
        relaxation="sigmoid",
        episodes=validation_episodes,
        validation_suite=validation_suite,
        return_rows=True,
    )
    mode_rows = build_mode_falsification_rows(policy, validation_rows, device=torch_device)
    case_rows = build_case_falsification_rows(validation_rows, mode_rows)
    falsification_summary = build_summary(mode_rows, case_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "ranker_falsification_mode_rows.csv",
        mode_rows,
        [
            "suite_index",
            "size",
            "seed",
            "discover_mode",
            "candidate_count",
            "chooser_selected_source",
            "predicted_source",
            "external_source",
            "chooser_selected_overlap",
            "predicted_overlap",
            "chooser_selected_pairs",
            "predicted_pairs",
            "chooser_selected_wirelength",
            "predicted_wirelength",
            "external_overlap",
            "external_pairs",
            "external_wirelength",
            "overlap_regret",
            "pair_regret",
            "wire_regret",
            "metric_match",
            "predicted_score",
            "external_score",
        ],
    )
    _write_csv(
        output_dir / "ranker_falsification_case_rows.csv",
        case_rows,
        [
            "suite_index",
            "size",
            "seed",
            "external_winning_mode",
            "external_source",
            "external_overlap",
            "external_pairs",
            "external_wirelength",
            "chooser_selected_winning_mode",
            "chooser_selected_source",
            "chooser_selected_overlap",
            "chooser_selected_pairs",
            "chooser_selected_wirelength",
            "predicted_winning_mode",
            "predicted_source",
            "predicted_overlap",
            "predicted_pairs",
            "predicted_wirelength",
            "overlap_regret",
            "pair_regret",
            "wire_regret",
            "metric_match",
        ],
    )
    payload = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "sizes": str(sizes_spec),
        "validation_episodes": int(validation_episodes),
        "validation_suite_seed": int(suite_seed),
        "temperature": float(temperature),
        "baseline_validation_summary": summary,
        "ranker_falsification_summary": falsification_summary,
        "output_files": {
            "mode_rows_csv": str(output_dir / "ranker_falsification_mode_rows.csv"),
            "case_rows_csv": str(output_dir / "ranker_falsification_case_rows.csv"),
        },
    }
    (output_dir / "ranker_falsification_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def infer_run_config(checkpoint, diagnosis_summary):
    sizes_spec = str(diagnosis_summary.get("sizes", default_sizes_from_checkpoint(checkpoint)))
    validation_episodes = int(
        diagnosis_summary.get("validation_episodes", default_validation_episodes_from_checkpoint(checkpoint))
    )
    suite_seed = int(diagnosis_summary.get("validation_suite_seed", default_suite_seed_from_checkpoint(checkpoint)))
    temperature = float(
        diagnosis_summary.get(
            "temperature",
            default_temperature_from_checkpoint(
                checkpoint,
                env_config_from_checkpoint(checkpoint, _arg_namespace_for_env()),
            ),
        )
    )
    return sizes_spec, validation_episodes, suite_seed, temperature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--diagnosis-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sizes", default=None)
    parser.add_argument("--validation-episodes", type=int, default=None)
    parser.add_argument("--validation-suite-seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    if args.diagnosis_dir is None and args.checkpoint is None:
        raise SystemExit("Provide either --diagnosis-dir or --checkpoint.")

    diagnosis_summary = {}
    if args.diagnosis_dir is not None:
        summary_path = args.diagnosis_dir / "diagnosis_summary.json"
        if summary_path.exists():
            diagnosis_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if args.checkpoint is None and diagnosis_summary.get("checkpoint"):
            args.checkpoint = Path(diagnosis_summary["checkpoint"])
        if args.output_dir is None:
            args.output_dir = args.diagnosis_dir
    if args.checkpoint is None:
        raise SystemExit("Could not infer checkpoint path.")
    if args.output_dir is None:
        args.output_dir = Path.cwd()

    device = args.device or default_device_arg()
    checkpoint_for_defaults = torch.load(args.checkpoint, map_location=device)
    sizes_spec, default_episodes, default_seed, default_temperature = infer_run_config(
        checkpoint_for_defaults,
        diagnosis_summary,
    )
    sizes_spec = args.sizes or sizes_spec or DEFAULT_SIZES
    validation_episodes = int(args.validation_episodes if args.validation_episodes is not None else default_episodes)
    suite_seed = int(args.validation_suite_seed if args.validation_suite_seed is not None else default_seed)
    temperature = float(args.temperature if args.temperature is not None else default_temperature)

    payload = run_falsification(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=device,
        sizes_spec=sizes_spec,
        validation_episodes=validation_episodes,
        suite_seed=suite_seed,
        temperature=temperature,
    )
    print(json.dumps(payload["ranker_falsification_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
