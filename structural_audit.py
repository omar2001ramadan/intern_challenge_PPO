"""Audit which decision structure is still missing from the RL system.

This module is intentionally stricter than ``wisdom_audit.py``.

The question here is not whether the codebase contains the name of an auxiliary
head. The question is where the benchmark-facing decision actually lives:

- inside the policy as a decisive choice
- in training as supervision that is not yet decisive
- in an outer wrapper / evaluator
- or nowhere at all
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


VALID_RL_TYPES = {"internal", "auxiliary_only", "external_wrapper", "missing"}
VALID_STRUCTURAL_STATUSES = {
    "internalized",
    "supervised_but_not_decisive",
    "externally_imposed",
    "missing",
}
VALID_PRIMARY_STRUCTURAL_GAPS = {
    "missing_internal_ranker",
    "missing_internal_cleanup_selector",
    "missing_internal_case_router",
    "missing_repair_value_model",
    "single_trajectory_bias",
    "missing_macro_context",
    "missing_locality_prior",
    "objective_mismatch",
    "none",
}
VALID_RELEVANCE = {"high", "medium", "low"}


def _csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_number(value):
    if value in ("", None):
        return value
    try:
        if any(ch in str(value) for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_diagnosis_artifacts(diagnosis_dir: Path):
    validation_rows = [
        {key: _parse_number(value) for key, value in row.items()}
        for row in _csv_rows(diagnosis_dir / "validation_case_ranking.csv")
    ]
    analyzed_rows = [
        {key: _parse_number(value) for key, value in row.items()}
        for row in _csv_rows(diagnosis_dir / "case_diagnosis.csv")
    ]
    return validation_rows, analyzed_rows


def repo_paths(repo_root: Path):
    return {
        "placement": repo_root / "placement.py",
        "test": repo_root / "test.py",
        "prior_solver": repo_root / "prior_solver.py",
        "teacher_solver": repo_root / "teacher_solver.py",
        "train_ppo": repo_root / "train_ppo.py",
        "env": repo_root / "env.py",
        "ordering_policy": repo_root / "ordering_policy.py",
        "run_diagnosis_suite": repo_root / "run_diagnosis_suite.py",
        "counterfactual_replay": repo_root / "counterfactual_replay.py",
    }


def _ref(path_key: str, token: str):
    return {"path_key": path_key, "token": token}


STRUCTURAL_NODE_SPECS = [
    {
        "decision_node": "case_family_routing",
        "benchmark_relevance": "high",
        "heuristic_has_node": True,
        "rl_has_node": True,
        "rl_type": "auxiliary_only",
        "structural_status": "supervised_but_not_decisive",
        "architecture_status": "present",
        "objective_status": "present",
        "scaffolding_status": "present",
        "heuristic_structure": "The benchmark-facing system enumerates discover modes and keeps the best case fit.",
        "rl_structure": "A mode-selector head exists, but the actual route winner still comes from portfolio enumeration in inference and validation.",
        "why_it_is_classified_this_way": "The model can score modes, but the benchmark-facing choice still depends on external multi-mode comparison.",
        "heuristic_refs": [
            _ref("placement", "DISCOVER_MODE_NAMES"),
            _ref("placement", "select_by_exact_overlap_then_wirelength"),
        ],
        "rl_arch_refs": [
            _ref("ordering_policy", "mode_selector_head"),
        ],
        "rl_objective_refs": [
            _ref("train_ppo", "mode_selector_top1"),
            _ref("train_ppo", "mode_selector_regret_overlap"),
        ],
        "rl_scaffold_refs": [
            _ref("placement", "discover_modes ="),
            _ref("placement", "saved.append"),
        ],
    },
    {
        "decision_node": "basin_generator_selection",
        "benchmark_relevance": "medium",
        "heuristic_has_node": True,
        "rl_has_node": True,
        "rl_type": "external_wrapper",
        "structural_status": "externally_imposed",
        "architecture_status": "missing",
        "objective_status": "missing",
        "scaffolding_status": "present",
        "heuristic_structure": "Different basin generators are tried and compared instead of trusting one rollout path.",
        "rl_structure": "The active basin generator is chosen by outer portfolio enumeration, not by an internal policy decision.",
        "why_it_is_classified_this_way": "The policy consumes a selected mode but does not itself choose which basin generator to run.",
        "heuristic_refs": [
            _ref("placement", "DISCOVER_MODE_NAMES"),
            _ref("placement", "for discover_mode in discover_modes"),
        ],
        "rl_arch_refs": [],
        "rl_objective_refs": [],
        "rl_scaffold_refs": [
            _ref("placement", "for discover_mode in discover_modes"),
            _ref("placement", "selected = select_by_exact_overlap_then_wirelength"),
        ],
    },
    {
        "decision_node": "legalization_repair_regime_choice",
        "benchmark_relevance": "low",
        "heuristic_has_node": True,
        "rl_has_node": True,
        "rl_type": "external_wrapper",
        "structural_status": "externally_imposed",
        "architecture_status": "present",
        "objective_status": "missing",
        "scaffolding_status": "present",
        "heuristic_structure": "Legalization and repair are distinct regimes with hard legality authority.",
        "rl_structure": "The environment exposes explicit phases, but regime boundaries are mainly encoded in the environment state machine.",
        "why_it_is_classified_this_way": "The separation exists structurally, but it is mostly imposed by the environment rather than learned as a benchmark-decisive decision.",
        "heuristic_refs": [
            _ref("placement", "select_by_exact_overlap_then_wirelength"),
        ],
        "rl_arch_refs": [
            _ref("env", "PlacementPhase"),
            _ref("env", "phase_step"),
        ],
        "rl_objective_refs": [],
        "rl_scaffold_refs": [
            _ref("env", "PHASE_DISCOVER"),
            _ref("env", "PHASE_REFINE"),
        ],
    },
    {
        "decision_node": "local_cleanup_operator_selection",
        "benchmark_relevance": "high",
        "heuristic_has_node": True,
        "rl_has_node": True,
        "rl_type": "auxiliary_only",
        "structural_status": "supervised_but_not_decisive",
        "architecture_status": "present",
        "objective_status": "present",
        "scaffolding_status": "present",
        "heuristic_structure": "Post-legal cleanup uses a portfolio of local operators and keeps the repaired winner.",
        "rl_structure": "A refine gate and cleanup head exist, but repaired variant wins are still adjudicated by the outer refine portfolio.",
        "why_it_is_classified_this_way": "The model can score cleanup variants, but the winner is still decided after wrapper-run repaired comparisons.",
        "heuristic_refs": [
            _ref("placement", "run_post_legal_refine_portfolio"),
            _ref("placement", "winning_refine_variant"),
        ],
        "rl_arch_refs": [
            _ref("ordering_policy", "cleanup_variant_head"),
        ],
        "rl_objective_refs": [
            _ref("train_ppo", "refine_gate_top1"),
            _ref("train_ppo", "cleanup_aux_loss"),
        ],
        "rl_scaffold_refs": [
            _ref("env", "run_post_legal_refine_portfolio"),
            _ref("placement", "winning_refine_variant"),
        ],
    },
    {
        "decision_node": "bounded_escape_decision",
        "benchmark_relevance": "low",
        "heuristic_has_node": True,
        "rl_has_node": True,
        "rl_type": "external_wrapper",
        "structural_status": "externally_imposed",
        "architecture_status": "missing",
        "objective_status": "partial",
        "scaffolding_status": "present",
        "heuristic_structure": "Temporary disorder is bounded and repaired before candidates are scored.",
        "rl_structure": "Unlock behavior exists as a phase/control regime, but the high-level decision to spend escape budget is still environment-scaffolded.",
        "why_it_is_classified_this_way": "The escape mechanism exists, but the current benchmark failures do not point to a missing learned escape chooser.",
        "heuristic_refs": [
            _ref("placement", "unlock"),
        ],
        "rl_arch_refs": [],
        "rl_objective_refs": [
            _ref("run_diagnosis_suite", "bounded_escape_missing"),
        ],
        "rl_scaffold_refs": [
            _ref("env", "unlock"),
        ],
    },
    {
        "decision_node": "repaired_candidate_selection",
        "benchmark_relevance": "high",
        "heuristic_has_node": True,
        "rl_has_node": True,
        "rl_type": "external_wrapper",
        "structural_status": "externally_imposed",
        "architecture_status": "present",
        "objective_status": "present",
        "scaffolding_status": "present",
        "heuristic_structure": "Candidates are scored after repair, then overlap-first selection keeps the repaired winner.",
        "rl_structure": "The policy has candidate-ranking auxiliaries, but repaired-candidate authority still lives in the evaluator and wrapper portfolio.",
        "why_it_is_classified_this_way": "The benchmark-facing winner is still selected outside the policy after repaired candidate audit.",
        "heuristic_refs": [
            _ref("placement", "select_by_exact_overlap_then_wirelength"),
            _ref("placement", "run_post_legal_refine_portfolio"),
        ],
        "rl_arch_refs": [
            _ref("ordering_policy", "candidate_rank_head"),
        ],
        "rl_objective_refs": [
            _ref("train_ppo", "ranking_aux_loss"),
            _ref("train_ppo", "repair_authority_agreement"),
        ],
        "rl_scaffold_refs": [
            _ref("train_ppo", "lexicographic_candidate_key"),
            _ref("placement", "selected = select_by_exact_overlap_then_wirelength"),
        ],
    },
    {
        "decision_node": "final_lexicographic_winner_selection",
        "benchmark_relevance": "high",
        "heuristic_has_node": True,
        "rl_has_node": True,
        "rl_type": "external_wrapper",
        "structural_status": "externally_imposed",
        "architecture_status": "missing",
        "objective_status": "present",
        "scaffolding_status": "present",
        "heuristic_structure": "The final winner is chosen by exact overlap first, then pair count, then wirelength.",
        "rl_structure": "Training sees lexicographic targets, but the benchmark-facing winner is still imposed by an external selector.",
        "why_it_is_classified_this_way": "This is intentionally allowed scaffolding, but it is not currently an internal RL decision structure.",
        "heuristic_refs": [
            _ref("placement", "select_by_exact_overlap_then_wirelength"),
        ],
        "rl_arch_refs": [],
        "rl_objective_refs": [
            _ref("train_ppo", "ranking_aux_loss"),
            _ref("train_ppo", "compare_episode_infos"),
        ],
        "rl_scaffold_refs": [
            _ref("placement", "selected = select_by_exact_overlap_then_wirelength"),
            _ref("train_ppo", "lexicographic_candidate_key"),
        ],
    },
    {
        "decision_node": "repair_value_estimation",
        "benchmark_relevance": "high",
        "heuristic_has_node": True,
        "rl_has_node": True,
        "rl_type": "auxiliary_only",
        "structural_status": "supervised_but_not_decisive",
        "architecture_status": "present",
        "objective_status": "present",
        "scaffolding_status": "present",
        "heuristic_structure": "The system values repaired candidates, not raw action proposals.",
        "rl_structure": "A candidate ranker and repair-authority supervision exist, but repaired value is still not the decisive internal object of control.",
        "why_it_is_classified_this_way": "The ranking target exists, yet the outer repaired-candidate comparison remains the true authority.",
        "heuristic_refs": [
            _ref("placement", "run_post_legal_refine_portfolio"),
        ],
        "rl_arch_refs": [
            _ref("ordering_policy", "candidate_rank_head"),
        ],
        "rl_objective_refs": [
            _ref("train_ppo", "repair_authority_agreement"),
            _ref("train_ppo", "candidate_records"),
        ],
        "rl_scaffold_refs": [
            _ref("train_ppo", "repaired_candidates"),
            _ref("placement", "score = dict(refine_result[\"score\"])"),
        ],
    },
    {
        "decision_node": "candidate_competition",
        "benchmark_relevance": "high",
        "heuristic_has_node": True,
        "rl_has_node": True,
        "rl_type": "external_wrapper",
        "structural_status": "externally_imposed",
        "architecture_status": "missing",
        "objective_status": "missing",
        "scaffolding_status": "present",
        "heuristic_structure": "The heuristic system compares multiple candidate paths before committing.",
        "rl_structure": "Candidate competition exists only in outer portfolios over modes and cleanup variants.",
        "why_it_is_classified_this_way": "The live policy still behaves like one trajectory at a time; the competition abstraction is wrapper-only.",
        "heuristic_refs": [
            _ref("placement", "saved = []"),
            _ref("placement", "saved.append"),
        ],
        "rl_arch_refs": [],
        "rl_objective_refs": [],
        "rl_scaffold_refs": [
            _ref("placement", "saved = []"),
            _ref("placement", "selected = select_by_exact_overlap_then_wirelength"),
        ],
    },
    {
        "decision_node": "stop_continue_revert_decision",
        "benchmark_relevance": "medium",
        "heuristic_has_node": True,
        "rl_has_node": True,
        "rl_type": "auxiliary_only",
        "structural_status": "supervised_but_not_decisive",
        "architecture_status": "present",
        "objective_status": "present",
        "scaffolding_status": "present",
        "heuristic_structure": "The search system preserves good basins and returns to a better candidate when continued rollout is clearly worse.",
        "rl_structure": "Continuation and rollback auxiliaries exist, but recent pilots show they are not yet benchmark-aligned.",
        "why_it_is_classified_this_way": "The decision exists in training logic, but it is not a reliable internal controller of benchmark outcomes.",
        "heuristic_refs": [
            _ref("placement", "best_candidate"),
        ],
        "rl_arch_refs": [
            _ref("ordering_policy", "continuation_preserve_head"),
        ],
        "rl_objective_refs": [
            _ref("train_ppo", "continuation_aux_loss"),
            _ref("train_ppo", "continuation_preserve_accuracy"),
        ],
        "rl_scaffold_refs": [
            _ref("env", "rollback_to_incumbent"),
            _ref("env", "best_so_far_returned"),
        ],
    },
    {
        "decision_node": "macro_context_use",
        "benchmark_relevance": "medium",
        "heuristic_has_node": True,
        "rl_has_node": True,
        "rl_type": "auxiliary_only",
        "structural_status": "supervised_but_not_decisive",
        "architecture_status": "present",
        "objective_status": "present",
        "scaffolding_status": "present",
        "heuristic_structure": "Macro-sensitive cases trigger different starts and refinement behavior.",
        "rl_structure": "Macro adjacency and case descriptor features exist, but the policy still relies on mode portfolio rescue on macro-sensitive cases.",
        "why_it_is_classified_this_way": "Macro context is represented, but not benchmark-decisive enough to replace external routing.",
        "heuristic_refs": [
            _ref("placement", "macro_clearance"),
            _ref("test", "num_macros"),
        ],
        "rl_arch_refs": [
            _ref("ordering_policy", "macro_adjacent"),
            _ref("ordering_policy", "mode_selector_head"),
        ],
        "rl_objective_refs": [
            _ref("train_ppo", "mode_selector_regret_overlap"),
        ],
        "rl_scaffold_refs": [
            _ref("placement", "discover_mode"),
        ],
    },
    {
        "decision_node": "locality_prior_use",
        "benchmark_relevance": "medium",
        "heuristic_has_node": True,
        "rl_has_node": True,
        "rl_type": "auxiliary_only",
        "structural_status": "supervised_but_not_decisive",
        "architecture_status": "present",
        "objective_status": "present",
        "scaffolding_status": "present",
        "heuristic_structure": "Same-size structure, translation-clean wire pressure, and local windows shape cleanup decisions.",
        "rl_structure": "Those features are exposed to cleanup auxiliaries, but the resulting local choice remains wrapper-dependent.",
        "why_it_is_classified_this_way": "The information exists in state, yet the benchmark-facing local decision is not internally decisive.",
        "heuristic_refs": [
            _ref("placement", "refine_window"),
        ],
        "rl_arch_refs": [
            _ref("ordering_policy", "translation_clean_wire_estimate"),
            _ref("ordering_policy", "same_size"),
        ],
        "rl_objective_refs": [
            _ref("train_ppo", "refine_gate_top1"),
        ],
        "rl_scaffold_refs": [
            _ref("env", "run_post_legal_refine_portfolio"),
        ],
    },
]


STRUCTURAL_GAP_SPECS = {
    "missing_internal_ranker": {
        "node": "repaired_candidate_selection",
        "heuristic_structure": "The heuristic promotes repaired candidates with overlap-first comparison before committing to the final answer.",
        "current_rl_location": "Candidate ranking exists as auxiliary supervision, but repaired-candidate authority is still enforced outside the policy.",
        "why_current_substitute_is_insufficient": "Continuation-style failures show the live policy does not preserve the earlier lexicographic winner on its own.",
        "falsification_experiment": "Run the fixed seven-case suite with the external lexicographic winner masked or replaced by the policy ranker and measure overlap/pair regret.",
        "smallest_next_change": "Introduce a decisive repaired-candidate ranker over a small candidate set from the same state, instead of ranking only after wrapper rescue.",
    },
    "missing_internal_cleanup_selector": {
        "node": "local_cleanup_operator_selection",
        "heuristic_structure": "The heuristic chooses among local cleanup operators by repaired outcome, not by one generic action path.",
        "current_rl_location": "The refine gate and cleanup auxiliaries exist, but the actual winner still comes from wrapper-run repaired comparisons.",
        "why_current_substitute_is_insufficient": "Low-overlap cases still default to incumbent_hold or brittle post-legal cleanup because the selector is not benchmark-decisive.",
        "falsification_experiment": "Restrict the wrapper to the gate's top-1 predicted cleanup variant and compare regret against the full repaired portfolio.",
        "smallest_next_change": "Turn the cleanup selector into a policy-level chooser whose value target is the repaired candidate, not the raw operator proposal.",
    },
    "missing_internal_case_router": {
        "node": "case_family_routing",
        "heuristic_structure": "The system routes cases toward the basin family that fits macro structure and congestion shape.",
        "current_rl_location": "A mode-selector head exists, but discover-mode success still comes from portfolio enumeration.",
        "why_current_substitute_is_insufficient": "Mode sensitivity is high across the fixed suite, so wrapper enumeration still carries routing quality.",
        "falsification_experiment": "Replace multi-mode evaluation with the learned mode selector on the full suite and measure regret versus portfolio routing.",
        "smallest_next_change": "Promote case routing to a first-class controller that chooses the discover regime before rollout.",
    },
    "missing_repair_value_model": {
        "node": "repair_value_estimation",
        "heuristic_structure": "The heuristic values post-repair candidates, which lets it reject raw moves that only look good before repair.",
        "current_rl_location": "Repair-authority metrics exist, but repaired value does not yet drive the main decision.",
        "why_current_substitute_is_insufficient": "Cleanup failures still show incumbent_hold winning because repaired quality is not the main learned control object.",
        "falsification_experiment": "Compare policy ranker outputs on repaired candidates against the external repaired winner and report agreement on the fixed suite.",
        "smallest_next_change": "Train a repaired-value head on candidate sets generated from the same incumbent and use it to score cleanup decisions before wrapper ranking.",
    },
    "single_trajectory_bias": {
        "node": "candidate_competition",
        "heuristic_structure": "The heuristic behaves like a search controller that compares candidates before committing.",
        "current_rl_location": "Candidate competition exists only as an outer portfolio over modes and refine variants.",
        "why_current_substitute_is_insufficient": "The policy still commits to one trajectory at a time, which exposes continuation failures after a good basin is found.",
        "falsification_experiment": "Compare single-trajectory execution against a small internal candidate set that is ranked before the next commit step.",
        "smallest_next_change": "Move from one-trajectory PPO control toward a small ranked candidate set per step or per phase.",
    },
}


STRUCTURAL_HYPOTHESIS_SPECS = [
    {
        "hypothesis_id": "H1",
        "structural_gap": "missing_internal_ranker",
        "claim": "RL lacks an internal legality-first ranker; winner selection is still mostly external.",
        "metric_name": "primary_structural_gap_count",
        "metric_target": "missing_internal_ranker",
        "failure_threshold": ">= 2 fixed-suite cases",
        "escape_condition": "If wrapper-free ranking preserves overlap-first winners with low regret on the fixed suite, the gap is not structural.",
        "experiment_definition": "Mask the external final selector and compare policy ranking versus lexicographic repaired winner.",
    },
    {
        "hypothesis_id": "H2",
        "structural_gap": "missing_internal_cleanup_selector",
        "claim": "RL lacks an internal post-legal operator chooser; refine wins still come mostly from wrapper competition.",
        "metric_name": "primary_structural_gap_count",
        "metric_target": "missing_internal_cleanup_selector",
        "failure_threshold": ">= 1 low-overlap fixed-suite case",
        "escape_condition": "If top-1 gate selection matches repaired winners with low regret, the gap is not structural.",
        "experiment_definition": "Run top-1 or top-2 gate-only cleanup against full refine portfolio and measure overlap/pair/wire regret.",
    },
    {
        "hypothesis_id": "H3",
        "structural_gap": "missing_internal_case_router",
        "claim": "RL lacks a real case router; discover-mode success still comes mostly from portfolio enumeration.",
        "metric_name": "mean_mode_sensitivity",
        "metric_target": "mode_sensitivity",
        "failure_threshold": ">= 0.50",
        "escape_condition": "If the learned router matches portfolio winners with low regret across the fixed suite, routing is not structurally missing.",
        "experiment_definition": "Replace per-mode portfolio enumeration with the learned mode selector and compare regret against the portfolio winner.",
    },
    {
        "hypothesis_id": "H4",
        "structural_gap": "missing_repair_value_model",
        "claim": "RL lacks a learned repair-value concept; it scores action proposals more than repaired candidates.",
        "metric_name": "secondary_structural_gap_count",
        "metric_target": "missing_repair_value_model",
        "failure_threshold": ">= 1 cleanup-sensitive case",
        "escape_condition": "If repaired-value predictions align with repaired winners and cleanup regret stays low, the gap is not structural.",
        "experiment_definition": "Score repaired candidates directly with the policy ranker and compare agreement with repaired lexicographic winners.",
    },
    {
        "hypothesis_id": "H5",
        "structural_gap": "single_trajectory_bias",
        "claim": "RL lacks internal candidate competition and therefore still behaves like one trajectory instead of a search controller.",
        "metric_name": "secondary_structural_gap_count",
        "metric_target": "single_trajectory_bias",
        "failure_threshold": ">= 2 continuation-sensitive cases",
        "escape_condition": "If a small internal candidate set does not outperform single-trajectory execution, the bias is not structural.",
        "experiment_definition": "Compare the current rollout against a small internally ranked candidate set built before commit steps.",
    },
]


PRIMARY_GAP_TO_NODE = {
    "missing_internal_ranker": "repaired_candidate_selection",
    "missing_internal_cleanup_selector": "local_cleanup_operator_selection",
    "missing_internal_case_router": "case_family_routing",
    "missing_repair_value_model": "repair_value_estimation",
    "single_trajectory_bias": "candidate_competition",
    "missing_macro_context": "macro_context_use",
    "missing_locality_prior": "locality_prior_use",
    "objective_mismatch": "stop_continue_revert_decision",
}


def _load_source_cache(repo_root: Path):
    paths = repo_paths(repo_root)
    cache = {}
    for key, path in paths.items():
        if path.exists():
            cache[key] = path.read_text(encoding="utf-8")
        else:
            cache[key] = ""
    return paths, cache


def _evidence_list(paths, cache, refs):
    evidence = []
    all_present = True
    for ref in refs:
        path_key = ref["path_key"]
        token = ref["token"]
        present = token in cache.get(path_key, "")
        all_present = all_present and present
        evidence.append(
            {
                "file": str(paths[path_key]),
                "token": token,
                "present": bool(present),
            }
        )
    return evidence, all_present


def _format_evidence(evidence):
    parts = []
    for item in evidence:
        status = "present" if item["present"] else "missing"
        parts.append(f"{Path(item['file']).name}:{item['token']} ({status})")
    return "; ".join(parts)


def build_structural_node_rows(repo_root: Path):
    paths, cache = _load_source_cache(repo_root)
    rows = []
    for spec in STRUCTURAL_NODE_SPECS:
        heuristic_evidence, heuristic_present = _evidence_list(paths, cache, spec["heuristic_refs"])
        rl_arch_evidence, arch_present = _evidence_list(paths, cache, spec["rl_arch_refs"])
        rl_objective_evidence, objective_present = _evidence_list(paths, cache, spec["rl_objective_refs"])
        rl_scaffold_evidence, scaffold_present = _evidence_list(paths, cache, spec["rl_scaffold_refs"])

        heuristic_has_node = bool(spec["heuristic_has_node"] and heuristic_present)
        rl_has_node = bool(spec["rl_has_node"] and (arch_present or objective_present or scaffold_present))
        if not rl_has_node:
            rl_type = "missing"
            structural_status = "missing"
        else:
            rl_type = spec["rl_type"]
            structural_status = spec["structural_status"]

        rows.append(
            {
                "decision_node": spec["decision_node"],
                "heuristic_has_node": bool(heuristic_has_node),
                "rl_has_node": bool(rl_has_node),
                "rl_type": rl_type,
                "structural_status": structural_status,
                "architecture_status": spec["architecture_status"],
                "objective_status": spec["objective_status"],
                "scaffolding_status": spec["scaffolding_status"],
                "benchmark_relevance": spec["benchmark_relevance"],
                "heuristic_structure": spec["heuristic_structure"],
                "rl_structure": spec["rl_structure"],
                "why_it_is_classified_this_way": spec["why_it_is_classified_this_way"],
                "heuristic_evidence": heuristic_evidence,
                "rl_architecture_evidence": rl_arch_evidence,
                "rl_objective_evidence": rl_objective_evidence,
                "rl_scaffolding_evidence": rl_scaffold_evidence,
                "evidence_files": _format_evidence(
                    heuristic_evidence + rl_arch_evidence + rl_objective_evidence + rl_scaffold_evidence
                ),
            }
        )
    return rows


def _as_float(row, key, default=0.0):
    value = row.get(key, default)
    if value in ("", None):
        return float(default)
    return float(value)


def _as_int(row, key, default=0):
    value = row.get(key, default)
    if value in ("", None):
        return int(default)
    return int(value)


def _heuristic_pattern(row):
    mode = str(row.get("winning_discover_mode", "unknown"))
    refine = str(row.get("winning_refine_variant", "unknown"))
    if refine == "incumbent_hold":
        cleanup = "preserve incumbent when repaired cleanup loses"
    else:
        cleanup = f"repair and keep local cleanup winner ({refine})"
    return f"portfolio route via {mode} -> {cleanup} -> lexicographic keep-best"


def _rl_pattern(row, primary_gap):
    mode = str(row.get("winning_discover_mode", "unknown"))
    refine = str(row.get("winning_refine_variant", "unknown"))
    if primary_gap == "missing_internal_ranker":
        return f"single rollout under {mode} -> later trajectory remains live -> external selector cleans up only at the end"
    if primary_gap == "missing_internal_cleanup_selector":
        return f"single rollout under {mode} -> cleanup wrapper tests variants -> {refine} survives because internal cleanup choice is weak"
    if primary_gap == "missing_internal_case_router":
        return f"discover route is obtained by mode enumeration around {mode}, not by an internal case router"
    if primary_gap == "missing_repair_value_model":
        return f"raw proposal path stays primary; repaired value is audited after the fact around {refine}"
    if primary_gap == "single_trajectory_bias":
        return f"one trajectory is committed under {mode}; candidate competition exists only outside the policy"
    return f"policy rollout under {mode} with wrapper-selected refine variant {refine}"


def _gap_from_case(validation_row, analyzed_row):
    factor = str(analyzed_row.get("dominant_factor", ""))
    winning_mode = str(validation_row.get("winning_discover_mode", "unknown"))
    winning_refine = str(validation_row.get("winning_refine_variant", "unknown"))
    mode_sensitivity = _as_float(validation_row, "mode_sensitivity")
    variant_sensitivity = _as_float(validation_row, "variant_sensitivity")

    if factor == "continuation_after_good_basin":
        primary = "missing_internal_ranker"
        secondary = "missing_internal_case_router" if winning_mode == "macro_clearance" and mode_sensitivity >= 0.85 else "single_trajectory_bias"
        divergence = "after a good repaired candidate exists, RL keeps one trajectory live instead of internally ranking and preserving the better candidate"
    elif factor in {"wire_recovery_missing", "local_variant_sensitivity"}:
        primary = "missing_internal_cleanup_selector"
        secondary = "missing_repair_value_model" if winning_refine == "incumbent_hold" else "missing_locality_prior"
        divergence = "once legality is strong enough, cleanup choice still depends on wrapper-run repaired comparisons instead of an internal selector"
    elif factor == "phr_budget_sensitivity":
        primary = "objective_mismatch"
        secondary = "missing_internal_case_router" if winning_mode == "macro_clearance" and mode_sensitivity >= 0.85 else "missing_repair_value_model"
        divergence = "the learned objective and inner repair budget are misaligned, so better repaired outcomes require external budget tuning"
    elif factor == "macro_refine_needed":
        primary = "missing_macro_context"
        secondary = "missing_internal_cleanup_selector"
        divergence = "macro-driven wire pressure remains outside the decisive internal cleanup logic"
    elif factor == "active_set_incompleteness":
        primary = "objective_mismatch"
        secondary = "missing_locality_prior"
        divergence = "training pressure is not aligned with full conflict exposure, so repair decisions rely on external audits"
    elif factor == "memory_sensitivity":
        primary = "objective_mismatch"
        secondary = "single_trajectory_bias"
        divergence = "rollout quality depends on trajectory-state quirks rather than a stable repaired-candidate decision structure"
    else:
        primary = "missing_internal_case_router" if mode_sensitivity >= 0.85 else "none"
        if primary == "missing_internal_case_router":
            secondary = "single_trajectory_bias"
            divergence = "routing quality still comes from external mode comparison"
        else:
            secondary = "none"
            divergence = "no structural divergence identified from the current diagnosis rows"

    if variant_sensitivity < 0.10 and primary == "missing_internal_cleanup_selector":
        secondary = "missing_internal_case_router" if winning_mode == "macro_clearance" else "none"

    return primary, secondary, divergence


def build_structural_case_rows(validation_rows, analyzed_rows):
    analyzed_by_case = {
        (int(row.get("suite_index", -1)), int(row.get("seed", -1))): row
        for row in analyzed_rows
    }
    rows = []
    for validation_row in validation_rows:
        suite_index = _as_int(validation_row, "suite_index", -1)
        seed = _as_int(validation_row, "seed", -1)
        analyzed_row = analyzed_by_case.get((suite_index, seed), {})
        primary, secondary, divergence = _gap_from_case(validation_row, analyzed_row)
        assert primary in VALID_PRIMARY_STRUCTURAL_GAPS
        assert secondary in VALID_PRIMARY_STRUCTURAL_GAPS
        evidence = (
            f"factor={analyzed_row.get('dominant_factor', 'none')} | "
            f"mode={validation_row.get('winning_discover_mode', 'unknown')} "
            f"(sens={_as_float(validation_row, 'mode_sensitivity'):.4f}) | "
            f"refine={validation_row.get('winning_refine_variant', 'unknown')} "
            f"(sens={_as_float(validation_row, 'variant_sensitivity'):.4f}) | "
            f"best={_as_float(validation_row, 'best_overlap'):.4f}/"
            f"{_as_int(validation_row, 'best_exact_overlap_pairs')}/"
            f"{_as_float(validation_row, 'best_wl'):.4f}"
        )
        rows.append(
            {
                "suite_index": suite_index,
                "size": str(validation_row.get("size", "")),
                "seed": seed,
                "winning_heuristic_decision_pattern": _heuristic_pattern(validation_row),
                "winning_rl_decision_pattern": _rl_pattern(validation_row, primary),
                "first_structural_divergence": divergence,
                "primary_structural_gap": primary,
                "secondary_structural_gap": secondary,
                "evidence": evidence,
            }
        )
    return rows


def build_structural_hypothesis_rows(case_rows, validation_rows, node_rows):
    primary_counts = Counter(row["primary_structural_gap"] for row in case_rows)
    secondary_counts = Counter(row["secondary_structural_gap"] for row in case_rows)
    mean_mode_sensitivity = sum(_as_float(row, "mode_sensitivity") for row in validation_rows) / max(len(validation_rows), 1)
    node_by_name = {row["decision_node"]: row for row in node_rows}

    rows = []
    for spec in STRUCTURAL_HYPOTHESIS_SPECS:
        if spec["metric_name"] == "primary_structural_gap_count":
            observed_value = int(primary_counts.get(spec["metric_target"], 0))
        elif spec["metric_name"] == "secondary_structural_gap_count":
            observed_value = int(secondary_counts.get(spec["metric_target"], 0))
        elif spec["metric_name"] == "mean_mode_sensitivity":
            observed_value = float(mean_mode_sensitivity)
        else:
            observed_value = 0

        node_row = node_by_name.get(STRUCTURAL_GAP_SPECS[spec["structural_gap"]]["node"])
        node_status = node_row["structural_status"] if node_row is not None else "missing"
        if spec["metric_name"] == "mean_mode_sensitivity":
            supported = bool(observed_value >= 0.50 and node_status != "internalized")
        else:
            supported = bool(observed_value and node_status != "internalized")

        rows.append(
            {
                "hypothesis_id": spec["hypothesis_id"],
                "structural_gap": spec["structural_gap"],
                "claim": spec["claim"],
                "metric_name": spec["metric_name"],
                "observed_value": observed_value,
                "failure_threshold": spec["failure_threshold"],
                "escape_condition": spec["escape_condition"],
                "experiment_definition": spec["experiment_definition"],
                "current_node_status": node_status,
                "supported_by_current_evidence": bool(supported),
            }
        )
    return rows


def build_structural_gap_report(node_rows, case_rows, hypothesis_rows):
    primary_counts = Counter(row["primary_structural_gap"] for row in case_rows)
    secondary_counts = Counter(row["secondary_structural_gap"] for row in case_rows)
    node_by_name = {row["decision_node"]: row for row in node_rows}

    high_impact = []
    for gap_id, count in primary_counts.most_common():
        if gap_id == "none":
            continue
        spec = STRUCTURAL_GAP_SPECS.get(gap_id)
        if spec is None:
            continue
        node_row = node_by_name[spec["node"]]
        high_impact.append(
            {
                "structural_gap": gap_id,
                "cases_flagged": int(count),
                "related_node": spec["node"],
                "node_rl_type": node_row["rl_type"],
                "node_structural_status": node_row["structural_status"],
                "heuristic_structure": spec["heuristic_structure"],
                "current_rl_location": spec["current_rl_location"],
                "why_current_substitute_is_insufficient": spec["why_current_substitute_is_insufficient"],
                "falsification_experiment": spec["falsification_experiment"],
                "smallest_next_change": spec["smallest_next_change"],
            }
        )

    wrapper_only = []
    for node in node_rows:
        if node["structural_status"] != "externally_imposed":
            continue
        if node["benchmark_relevance"] != "high":
            continue
        wrapper_only.append(
            {
                "decision_node": node["decision_node"],
                "benchmark_relevance": node["benchmark_relevance"],
                "rl_type": node["rl_type"],
                "heuristic_structure": node["heuristic_structure"],
                "rl_structure": node["rl_structure"],
                "why_wrapper_only": node["why_it_is_classified_this_way"],
            }
        )

    not_missing = []
    for node in node_rows:
        primary_count = int(primary_counts.get(node["decision_node"], 0))
        if node["benchmark_relevance"] == "low" or node["decision_node"] in {"bounded_escape_decision", "legalization_repair_regime_choice"}:
            not_missing.append(
                {
                    "decision_node": node["decision_node"],
                    "benchmark_relevance": node["benchmark_relevance"],
                    "structural_status": node["structural_status"],
                    "why_not_currently_primary": "Current fixed-suite evidence does not flag this structure as a dominant failure source.",
                }
            )
        elif primary_count == 0 and node["decision_node"] in {"macro_context_use", "locality_prior_use", "stop_continue_revert_decision"}:
            not_missing.append(
                {
                    "decision_node": node["decision_node"],
                    "benchmark_relevance": node["benchmark_relevance"],
                    "structural_status": node["structural_status"],
                    "why_not_currently_primary": "The structure may be imperfect, but the current seven-case suite does not rank it as the main failure frontier.",
                }
            )

    top_gap = high_impact[0] if high_impact else None
    next_experiment = None
    if top_gap is not None:
        next_experiment = {
            "priority_gap": top_gap["structural_gap"],
            "goal": top_gap["smallest_next_change"],
            "falsification_experiment": top_gap["falsification_experiment"],
        }

    summary = {
        "primary_structural_gap_counts": dict(sorted(primary_counts.items())),
        "secondary_structural_gap_counts": dict(sorted(secondary_counts.items())),
        "rl_type_counts": dict(sorted(Counter(row["rl_type"] for row in node_rows).items())),
        "structural_status_counts": dict(sorted(Counter(row["structural_status"] for row in node_rows).items())),
        "supported_hypothesis_count": int(sum(1 for row in hypothesis_rows if row["supported_by_current_evidence"])),
    }
    return {
        "summary": summary,
        "structurally_missing_high_impact": high_impact,
        "present_only_through_scaffolding": wrapper_only,
        "not_actually_missing": not_missing,
        "next_architecture_experiment": next_experiment,
    }


def _report_markdown(report):
    lines = ["# Structural Gap Report", ""]
    lines.append("## Structurally Missing And High Impact")
    if report["structurally_missing_high_impact"]:
        for row in report["structurally_missing_high_impact"]:
            lines.append(f"- `{row['structural_gap']}` on {row['cases_flagged']} fixed-suite cases.")
            lines.append(f"  Heuristic structure: {row['heuristic_structure']}")
            lines.append(f"  RL today: {row['current_rl_location']}")
            lines.append(f"  Why insufficient: {row['why_current_substitute_is_insufficient']}")
            lines.append(f"  Next experiment: {row['falsification_experiment']}")
            lines.append(f"  Smallest next change: {row['smallest_next_change']}")
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("## Present Only Through Scaffolding")
    if report["present_only_through_scaffolding"]:
        for row in report["present_only_through_scaffolding"]:
            lines.append(f"- `{row['decision_node']}` remains wrapper-only: {row['why_wrapper_only']}")
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("## Not Actually Missing")
    if report["not_actually_missing"]:
        for row in report["not_actually_missing"]:
            lines.append(f"- `{row['decision_node']}` is not the current frontier: {row['why_not_currently_primary']}")
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("## Next Architecture Experiment")
    if report["next_architecture_experiment"] is not None:
        item = report["next_architecture_experiment"]
        lines.append(f"- Priority gap: `{item['priority_gap']}`")
        lines.append(f"- Goal: {item['goal']}")
        lines.append(f"- Falsification experiment: {item['falsification_experiment']}")
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def write_structural_outputs(*, repo_root: Path, diagnosis_dir: Path, validation_rows, analyzed_rows):
    node_rows = build_structural_node_rows(repo_root)
    case_rows = build_structural_case_rows(validation_rows, analyzed_rows)
    hypothesis_rows = build_structural_hypothesis_rows(case_rows, validation_rows, node_rows)
    report = build_structural_gap_report(node_rows, case_rows, hypothesis_rows)

    node_csv_rows = [
        {
            "decision_node": row["decision_node"],
            "heuristic_has_node": row["heuristic_has_node"],
            "rl_has_node": row["rl_has_node"],
            "rl_type": row["rl_type"],
            "structural_status": row["structural_status"],
            "architecture_status": row["architecture_status"],
            "objective_status": row["objective_status"],
            "scaffolding_status": row["scaffolding_status"],
            "benchmark_relevance": row["benchmark_relevance"],
            "evidence_files": row["evidence_files"],
        }
        for row in node_rows
    ]
    hypothesis_csv_rows = [
        {
            "hypothesis_id": row["hypothesis_id"],
            "structural_gap": row["structural_gap"],
            "metric_name": row["metric_name"],
            "observed_value": row["observed_value"],
            "failure_threshold": row["failure_threshold"],
            "current_node_status": row["current_node_status"],
            "supported_by_current_evidence": row["supported_by_current_evidence"],
        }
        for row in hypothesis_rows
    ]

    _write_csv(
        diagnosis_dir / "structural_node_table.csv",
        node_csv_rows,
        [
            "decision_node",
            "heuristic_has_node",
            "rl_has_node",
            "rl_type",
            "structural_status",
            "architecture_status",
            "objective_status",
            "scaffolding_status",
            "benchmark_relevance",
            "evidence_files",
        ],
    )
    _write_csv(
        diagnosis_dir / "structural_hypotheses.csv",
        hypothesis_csv_rows,
        [
            "hypothesis_id",
            "structural_gap",
            "metric_name",
            "observed_value",
            "failure_threshold",
            "current_node_status",
            "supported_by_current_evidence",
        ],
    )
    _write_csv(
        diagnosis_dir / "structural_case_gaps.csv",
        case_rows,
        [
            "suite_index",
            "size",
            "seed",
            "winning_heuristic_decision_pattern",
            "winning_rl_decision_pattern",
            "first_structural_divergence",
            "primary_structural_gap",
            "secondary_structural_gap",
            "evidence",
        ],
    )

    (diagnosis_dir / "structural_node_table.json").write_text(
        json.dumps(node_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (diagnosis_dir / "structural_hypotheses.json").write_text(
        json.dumps(hypothesis_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (diagnosis_dir / "structural_gap_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (diagnosis_dir / "structural_gap_report.md").write_text(
        _report_markdown(report),
        encoding="utf-8",
    )

    return {
        "node_rows": node_rows,
        "case_rows": case_rows,
        "hypothesis_rows": hypothesis_rows,
        "report": report,
        "output_files": {
            "structural_node_table_csv": str(diagnosis_dir / "structural_node_table.csv"),
            "structural_node_table_json": str(diagnosis_dir / "structural_node_table.json"),
            "structural_hypotheses_csv": str(diagnosis_dir / "structural_hypotheses.csv"),
            "structural_hypotheses_json": str(diagnosis_dir / "structural_hypotheses.json"),
            "structural_case_gaps_csv": str(diagnosis_dir / "structural_case_gaps.csv"),
            "structural_gap_report_json": str(diagnosis_dir / "structural_gap_report.json"),
            "structural_gap_report_md": str(diagnosis_dir / "structural_gap_report.md"),
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnosis-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    diagnosis_dir = Path(args.diagnosis_dir)
    validation_rows, analyzed_rows = load_diagnosis_artifacts(diagnosis_dir)
    result = write_structural_outputs(
        repo_root=repo_root,
        diagnosis_dir=diagnosis_dir,
        validation_rows=validation_rows,
        analyzed_rows=analyzed_rows,
    )
    print(json.dumps(result["report"]["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
