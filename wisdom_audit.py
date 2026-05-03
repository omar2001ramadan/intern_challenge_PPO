"""Audit which heuristic wisdoms are internalized by the RL system.

This module intentionally does not compare operator-by-operator. The goal is to
describe which higher-level decision priors from the benchmark-facing heuristic
have been represented, optimized, separated, selected, and generalized in the
current RL stack.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


VALID_DIMENSION_STATUSES = {"full", "partial", "none"}
VALID_WISDOM_STATUSES = {
    "internalized",
    "partially_internalized",
    "externally_approximated",
    "missing",
}
VALID_TRANSFER_MECHANISMS = {
    "State augmentation",
    "Auxiliary supervision",
    "Ranking / reward restructuring",
    "Curriculum",
    "Architecture separation",
    "Validation-time portfolio only",
    "Diagnosis only",
}
VALID_TRANSFER_BUCKETS = {
    "should_be_learned_inside_rl",
    "should_remain_outer_loop_scaffolding",
    "should_not_be_transferred",
}

WISDOM_LABELS = {
    "legality_first_selection_missing",
    "candidate_competition_missing",
    "case_shape_routing_missing",
    "locality_control_missing",
    "bounded_escape_missing",
    "post_legal_cleanup_bias_missing",
    "macro_context_missing",
    "credit_assignment_too_diffuse",
    "constraint_exposure_missing",
    "none",
}


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
        "heuristic_public": repo_root / "placement.py",
        "benchmark_harness": repo_root / "test.py",
        "heuristic_alias": repo_root / "prior_solver.py",
        "teacher_path": repo_root / "teacher_solver.py",
        "rl_training": repo_root / "train_ppo.py",
        "rl_env": repo_root / "env.py",
        "rl_policy": repo_root / "ordering_policy.py",
        "diagnosis": repo_root / "run_diagnosis_suite.py",
        "counterfactual": repo_root / "counterfactual_replay.py",
    }


def _ref(file_key: str, token: str):
    return {"file_key": file_key, "token": token}


WISDOM_CLASS_SPECS = [
    {
        "wisdom_class": "stage_basin_formation",
        "group": "Stage wisdom",
        "decision_it_supports": "Choose an early geometry basin that legalization can finish instead of rescuing a bad start later.",
        "document_claim": "The solver improved in stages: first remove collisions, then shorten wires, then escape stuck layouts, then choose the best rule per case.",
        "heuristic_evidence": "Document sections 2, 3, and 7 emphasize spread-first and macro-shape-sensitive starts.",
        "represented": "full",
        "optimized": "partial",
        "separated": "full",
        "selected": "partial",
        "generalized": "partial",
        "wisdom_status": "partially_internalized",
        "transfer_mechanism": "Architecture separation",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Keep phase-separated DISCOVER, but learn stronger basin-quality prediction instead of relying on post hoc mode wins.",
        "success_metric": "The same discover mode wins because of learned case-fit, not because evaluation portfolio rescues bad alternatives.",
        "heuristic_refs": [
            _ref("heuristic_public", "mode"),
            _ref("benchmark_harness", "spread_radius"),
        ],
        "rl_refs": [
            _ref("rl_training", "discover_mode"),
            _ref("rl_env", "PHASE_DISCOVER"),
            _ref("rl_policy", "DISCOVER_MODE_NAMES"),
        ],
    },
    {
        "wisdom_class": "stage_legalization",
        "group": "Stage wisdom",
        "decision_it_supports": "Focus on legality as a distinct regime before wirelength cleanup.",
        "document_claim": "A zero-overlap candidate beats any overlapping candidate.",
        "heuristic_evidence": "Document sections 3 and 6 make legality the hard pass/fail rule.",
        "represented": "full",
        "optimized": "partial",
        "separated": "full",
        "selected": "full",
        "generalized": "partial",
        "wisdom_status": "partially_internalized",
        "transfer_mechanism": "Ranking / reward restructuring",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Move legality-first selection pressure from only evaluation into value/ranking targets during training.",
        "success_metric": "Validation and rollout best-so-far candidates agree more often on overlap-first ordering.",
        "heuristic_refs": [
            _ref("benchmark_harness", "num_cells_with_overlaps"),
        ],
        "rl_refs": [
            _ref("rl_env", "_stop_reward"),
            _ref("rl_training", "best_exact_overlap_pairs"),
            _ref("diagnosis", "best_overlap"),
        ],
    },
    {
        "wisdom_class": "stage_post_legal_cleanup",
        "group": "Stage wisdom",
        "decision_it_supports": "Once legality is strong enough, shift to safe local wirelength recovery rather than reopening global search.",
        "document_claim": "Local refinement starts after the solver has a legal placement. The goal is to shorten wires without breaking the no-overlap rule.",
        "heuristic_evidence": "Document section 4 describes local cleanup after legality.",
        "represented": "full",
        "optimized": "partial",
        "separated": "full",
        "selected": "full",
        "generalized": "partial",
        "wisdom_status": "externally_approximated",
        "transfer_mechanism": "Architecture separation",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Train explicit post-legal value/ranking bias so refine variants are not only chosen by outer-loop comparison.",
        "success_metric": "Non-hold refine variants win consistently without relying only on validation-time selection.",
        "heuristic_refs": [
            _ref("heuristic_public", "select_by_exact_overlap_then_wirelength"),
        ],
        "rl_refs": [
            _ref("rl_env", "winning_refine_variant"),
            _ref("rl_training", "post_legal_refine_portfolio"),
            _ref("rl_policy", "cleanup_variant_head"),
            _ref("rl_training", "cleanup_aux_loss"),
            _ref("heuristic_public", "winning_refine_variant"),
        ],
    },
    {
        "wisdom_class": "stage_stuck_layout_escape",
        "group": "Stage wisdom",
        "decision_it_supports": "Escape a stuck legal or near-legal state using bounded temporary disorder followed by repair.",
        "document_claim": "A messy intermediate state is allowed. A messy final state is never allowed.",
        "heuristic_evidence": "Document section 5 defines unlock windows and mandatory relegalization.",
        "represented": "full",
        "optimized": "partial",
        "separated": "full",
        "selected": "partial",
        "generalized": "partial",
        "wisdom_status": "partially_internalized",
        "transfer_mechanism": "Auxiliary supervision",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Learn when unlock is worth attempting and whether repaired candidates will improve before paying the exploration cost.",
        "success_metric": "Unlock attempts become rarer but more accepted, with fewer legality-breaking repairs.",
        "heuristic_refs": [
            _ref("heuristic_public", "update_active_pair_cache"),
        ],
        "rl_refs": [
            _ref("rl_env", "unlock"),
            _ref("diagnosis", "bounded_escape_missing"),
        ],
    },
    {
        "wisdom_class": "selection_legality_first_ranking",
        "group": "Selection wisdom",
        "decision_it_supports": "Select winners by overlap first, then exact overlap pairs, then wirelength.",
        "document_claim": "The rule that keeps the solver honest is zero-overlap first, then lower wirelength.",
        "heuristic_evidence": "Document sections 3 and 6 define the outer contract.",
        "represented": "full",
        "optimized": "partial",
        "separated": "full",
        "selected": "full",
        "generalized": "partial",
        "wisdom_status": "externally_approximated",
        "transfer_mechanism": "Ranking / reward restructuring",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Teach lexicographic legality-first ranking with auxiliary pairwise ranking or value heads instead of depending on evaluation wrappers.",
        "success_metric": "Policy-preferred trajectories align with the external lexicographic selector more often.",
        "heuristic_refs": [
            _ref("heuristic_public", "select_by_exact_overlap_then_wirelength"),
        ],
        "rl_refs": [
            _ref("rl_training", "compare_episode_infos"),
            _ref("rl_policy", "candidate_rank_head"),
            _ref("rl_training", "ranking_aux_loss"),
            _ref("heuristic_public", "winning_refine_variant"),
        ],
    },
    {
        "wisdom_class": "selection_do_not_keep_messy_states",
        "group": "Selection wisdom",
        "decision_it_supports": "Never preserve a candidate that remains messy after repair.",
        "document_claim": "Do-not-keep messy states; repaired legal candidates only.",
        "heuristic_evidence": "Document section 5 states messy final states are never kept.",
        "represented": "full",
        "optimized": "partial",
        "separated": "full",
        "selected": "full",
        "generalized": "partial",
        "wisdom_status": "externally_approximated",
        "transfer_mechanism": "Auxiliary supervision",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Supervise repair-acceptance prediction so the policy learns to avoid unrecoverable proposals.",
        "success_metric": "Rejected refine and unlock proposals drop while legality stays stable.",
        "heuristic_refs": [
            _ref("heuristic_public", "select_by_exact_overlap_then_wirelength"),
        ],
        "rl_refs": [
            _ref("rl_env", "rollback_to_incumbent"),
            _ref("rl_env", "refine_variant_repair_legal"),
        ],
    },
    {
        "wisdom_class": "selection_candidate_competition",
        "group": "Selection wisdom",
        "decision_it_supports": "Compare multiple candidates instead of trusting one trajectory family.",
        "document_claim": "The final solver is a portfolio.",
        "heuristic_evidence": "Document sections 3 and 6 emphasize candidate competition.",
        "represented": "partial",
        "optimized": "none",
        "separated": "full",
        "selected": "full",
        "generalized": "partial",
        "wisdom_status": "externally_approximated",
        "transfer_mechanism": "Validation-time portfolio only",
        "transfer_bucket": "should_remain_outer_loop_scaffolding",
        "transfer_path": "Keep evaluation-time candidate competition, but do not confuse this with learned policy wisdom.",
        "success_metric": "Per-case winner diversity is stable, while training no longer depends on one rescue portfolio to look strong.",
        "heuristic_refs": [
            _ref("benchmark_harness", "run_all_tests"),
        ],
        "rl_refs": [
            _ref("rl_training", "for discover_mode in DISCOVER_MODE_NAMES"),
            _ref("heuristic_public", "winning_refine_variant"),
        ],
    },
    {
        "wisdom_class": "locality_global_vs_local",
        "group": "Locality wisdom",
        "decision_it_supports": "Use global moves for basin formation and local moves for cleanup instead of one blended action family.",
        "document_claim": "Temporary unlock windows, legal projection, and local cleanup are distinct from global placement pressure.",
        "heuristic_evidence": "Document sections 3, 4, and 5 separate global search from local refinement.",
        "represented": "full",
        "optimized": "partial",
        "separated": "full",
        "selected": "partial",
        "generalized": "partial",
        "wisdom_status": "partially_internalized",
        "transfer_mechanism": "Architecture separation",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Keep phase-local heads and teach stronger local objective structure so locality is not only enforced by wrappers.",
        "success_metric": "Refine variants improve wirelength more often without reopening basin search behavior.",
        "heuristic_refs": [
            _ref("heuristic_public", "train_placement"),
        ],
        "rl_refs": [
            _ref("rl_env", "PHASE_REFINE"),
            _ref("rl_policy", "phase_step"),
        ],
    },
    {
        "wisdom_class": "locality_same_size_structure",
        "group": "Locality wisdom",
        "decision_it_supports": "Exploit same-size equivalence when swapping or reassigning legal slots.",
        "document_claim": "If standard cells have the same size, they can trade legal slots.",
        "heuristic_evidence": "Document section 4 explicitly names same-size assignment.",
        "represented": "partial",
        "optimized": "partial",
        "separated": "partial",
        "selected": "partial",
        "generalized": "none",
        "wisdom_status": "partially_internalized",
        "transfer_mechanism": "State augmentation",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Expose same-size group structure directly to local refinement scoring and ranking heads.",
        "success_metric": "Same-size-aware refine variants win on the cases where slot reassignment is currently the best local operator.",
        "heuristic_refs": [
            _ref("heuristic_public", "swap_or_reassign_local"),
        ],
        "rl_refs": [
            _ref("rl_env", "swap_or_reassign_local"),
        ],
    },
    {
        "wisdom_class": "locality_translation_clean_scoring",
        "group": "Locality wisdom",
        "decision_it_supports": "Avoid attributing wirelength gains to moves that only translate both pins of the same cell together.",
        "document_claim": "Translation-clean scoring helped some cases.",
        "heuristic_evidence": "Document sections 4 and 7 describe translation-clean scoring as case-dependent.",
        "represented": "none",
        "optimized": "none",
        "separated": "none",
        "selected": "none",
        "generalized": "none",
        "wisdom_status": "missing",
        "transfer_mechanism": "State augmentation",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Provide translation-clean wirelength features or auxiliary targets so local scoring can ignore same-cell invariants when appropriate.",
        "success_metric": "Dense-detail cases stop showing fake local wire pressure that does not improve true wirelength.",
        "heuristic_refs": [
            _ref("benchmark_harness", "normalized_wl"),
        ],
        "rl_refs": [
            _ref("rl_training", "normalized_wl"),
            _ref("rl_env", "translation_clean"),
            _ref("rl_env", "cleanup_feature_vector"),
        ],
    },
    {
        "wisdom_class": "locality_partial_move_conservatism",
        "group": "Locality wisdom",
        "decision_it_supports": "Prefer smaller safe moves when full moves destabilize legality or local wire recovery.",
        "document_claim": "Do not always move a cell all the way to its target. Sometimes a smaller step is safer and scores better.",
        "heuristic_evidence": "Document section 4 defines a partial-move gate.",
        "represented": "partial",
        "optimized": "partial",
        "separated": "partial",
        "selected": "partial",
        "generalized": "partial",
        "wisdom_status": "partially_internalized",
        "transfer_mechanism": "Auxiliary supervision",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Teach a move-size acceptance bias in cleanup stages rather than only clipping residuals heuristically.",
        "success_metric": "Accepted local variants show lower legality break rate at similar wirelength gain.",
        "heuristic_refs": [
            _ref("heuristic_public", "step_scale"),
        ],
        "rl_refs": [
            _ref("rl_env", "step_scale"),
            _ref("counterfactual", "step_scale_delta"),
        ],
    },
    {
        "wisdom_class": "case_shape_routing",
        "group": "Case-shape wisdom",
        "decision_it_supports": "Route cases toward different strategies based on shape and congestion pattern.",
        "document_claim": "The solver chooses the best rule for each test case.",
        "heuristic_evidence": "Document section 7 is a case-by-case gate profile.",
        "represented": "full",
        "optimized": "partial",
        "separated": "partial",
        "selected": "full",
        "generalized": "partial",
        "wisdom_status": "externally_approximated",
        "transfer_mechanism": "Auxiliary supervision",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Replace fixed discover-mode enumeration with a learned case-family router trained from case descriptors and mode outcomes.",
        "success_metric": "A learned mode selector predicts the winning discover mode above the balanced baseline.",
        "heuristic_refs": [
            _ref("benchmark_harness", "TEST_CASES"),
        ],
        "rl_refs": [
            _ref("rl_training", "mode_win_counts"),
            _ref("rl_policy", "mode_selector_head"),
            _ref("rl_training", "mode_selector_top1"),
            _ref("diagnosis", "winning_discover_mode"),
        ],
    },
    {
        "wisdom_class": "case_shape_macro_context",
        "group": "Case-shape wisdom",
        "decision_it_supports": "Handle macro-dominated cases differently because large cells shape the feasible basin.",
        "document_claim": "Large cells decide where many small cells can fit.",
        "heuristic_evidence": "Document sections 6 and 7 emphasize macro-sensitive search.",
        "represented": "partial",
        "optimized": "partial",
        "separated": "partial",
        "selected": "partial",
        "generalized": "partial",
        "wisdom_status": "partially_internalized",
        "transfer_mechanism": "State augmentation",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Expose macro-adjacency and macro-driven wire gradient structure directly to routing and cleanup decisions.",
        "success_metric": "Macro-heavy cases no longer require outer-loop macro-specific rescues to match legality basins.",
        "heuristic_refs": [
            _ref("benchmark_harness", "num_macros"),
        ],
        "rl_refs": [
            _ref("rl_training", "macro_clearance"),
            _ref("diagnosis", "macro_refine_needed"),
        ],
    },
    {
        "wisdom_class": "case_shape_large_scale_pruning",
        "group": "Case-shape wisdom",
        "decision_it_supports": "Prune pair checking and localize constraints differently at very large scales.",
        "document_claim": "Scale matters most. Every pair cannot be checked naively.",
        "heuristic_evidence": "Document sections 3 and 7 emphasize KD-tree pruning and large-design gates.",
        "represented": "full",
        "optimized": "partial",
        "separated": "partial",
        "selected": "partial",
        "generalized": "partial",
        "wisdom_status": "partially_internalized",
        "transfer_mechanism": "State augmentation",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Keep scalable constraint exposure, but teach the policy which local signals stay reliable as size grows.",
        "success_metric": "Large-scale cases stop degrading mainly due to diffuse or incomplete constraint exposure.",
        "heuristic_refs": [
            _ref("heuristic_public", "build_initial_active_pairs"),
        ],
        "rl_refs": [
            _ref("counterfactual", "active_set_all_pairs"),
            _ref("rl_env", "active_pairs"),
        ],
    },
    {
        "wisdom_class": "search_preserve_good_basins",
        "group": "Search-discipline wisdom",
        "decision_it_supports": "Preserve incumbents and revert after regressions instead of trusting continuation.",
        "document_claim": "Keep a stabilizing pull toward a previous good position when that stabilizer improves the measured result.",
        "heuristic_evidence": "Document section 4 names an anchor gate and section 3 defines keep-the-best selection.",
        "represented": "full",
        "optimized": "partial",
        "separated": "full",
        "selected": "full",
        "generalized": "partial",
        "wisdom_status": "partially_internalized",
        "transfer_mechanism": "Auxiliary supervision",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Turn rollback and incumbent choice into explicit learned preference signals rather than only environment-side corrections.",
        "success_metric": "Continuation failure stops being a dominant diagnosis without relying on hard rollback alone.",
        "heuristic_refs": [
            _ref("heuristic_public", "select_by_exact_overlap_then_wirelength"),
        ],
        "rl_refs": [
            _ref("rl_env", "rollback_to_incumbent"),
            _ref("diagnosis", "continuation_after_good_basin"),
        ],
    },
    {
        "wisdom_class": "search_bounded_escape",
        "group": "Search-discipline wisdom",
        "decision_it_supports": "Escape only in bounded regions instead of reopening the whole layout.",
        "document_claim": "Pick a small hot region, allow overlap there during search, then repair the region.",
        "heuristic_evidence": "Document section 5 defines temporary unlock windows.",
        "represented": "full",
        "optimized": "partial",
        "separated": "full",
        "selected": "partial",
        "generalized": "partial",
        "wisdom_status": "partially_internalized",
        "transfer_mechanism": "Architecture separation",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Keep bounded unlock as structure, but learn hotspot selection and repair-worthiness.",
        "success_metric": "Unlock windows are accepted more often and stop behaving like random local noise.",
        "heuristic_refs": [
            _ref("heuristic_public", "update_active_pair_cache"),
        ],
        "rl_refs": [
            _ref("rl_env", "unlock_window"),
            _ref("rl_env", "unlock_steps_remaining"),
        ],
    },
    {
        "wisdom_class": "search_relegalize_before_scoring",
        "group": "Search-discipline wisdom",
        "decision_it_supports": "Repair and audit before deciding whether a candidate helps.",
        "document_claim": "During repair, the solver does not simply put cells in the nearest legal spot; it scores candidate spots by whether they shorten important wires.",
        "heuristic_evidence": "Document section 5 couples unlock with relegalization and scoring.",
        "represented": "full",
        "optimized": "partial",
        "separated": "full",
        "selected": "full",
        "generalized": "partial",
        "wisdom_status": "externally_approximated",
        "transfer_mechanism": "Auxiliary supervision",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Learn a repair-quality predictor so the policy internalizes that repaired legality is the scoring authority.",
        "success_metric": "Candidate acceptance correlates more strongly with predicted repaired metrics than with raw proposal metrics.",
        "heuristic_refs": [
            _ref("heuristic_public", "calculate_normalized_metrics"),
        ],
        "rl_refs": [
            _ref("rl_env", "exact_audit"),
            _ref("rl_env", "refine_variant_repair_legal"),
        ],
    },
    {
        "wisdom_class": "search_separate_exploration_from_cleanup",
        "group": "Search-discipline wisdom",
        "decision_it_supports": "Separate exploratory basin moves from conservative cleanup moves.",
        "document_claim": "The solver improved in stages and kept different cleanup rules for different case shapes.",
        "heuristic_evidence": "Document sections 2, 4, and 7 separate exploration from cleanup gates.",
        "represented": "full",
        "optimized": "partial",
        "separated": "full",
        "selected": "partial",
        "generalized": "partial",
        "wisdom_status": "partially_internalized",
        "transfer_mechanism": "Architecture separation",
        "transfer_bucket": "should_be_learned_inside_rl",
        "transfer_path": "Keep phase separation, but learn stage-aware objectives so cleanup no longer depends mostly on external portfolios.",
        "success_metric": "Cleanup-stage actions improve final lexicographic score without reintroducing DISCOVER-like diversity.",
        "heuristic_refs": [
            _ref("heuristic_public", "mode"),
        ],
        "rl_refs": [
            _ref("rl_env", "phase"),
            _ref("rl_policy", "PHASE_REQUEST_TO_INDEX"),
        ],
    },
]


def _file_contains(path: Path, token: str) -> bool:
    if not path.exists():
        return False
    try:
        return token in path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False


def _reference_status(repo_root: Path, refs):
    paths = repo_paths(repo_root)
    statuses = []
    for ref in refs:
        path = paths[ref["file_key"]]
        statuses.append(
            {
                "file": str(path),
                "token": ref["token"],
                "present": bool(_file_contains(path, ref["token"])),
            }
        )
    return statuses


def _validate_spec(spec):
    for key in ("represented", "optimized", "separated", "selected", "generalized"):
        if spec[key] not in VALID_DIMENSION_STATUSES:
            raise ValueError(f"Invalid dimension status for {spec['wisdom_class']}: {key}={spec[key]}")
    if spec["wisdom_status"] not in VALID_WISDOM_STATUSES:
        raise ValueError(f"Invalid wisdom status for {spec['wisdom_class']}")
    if spec["transfer_mechanism"] not in VALID_TRANSFER_MECHANISMS:
        raise ValueError(f"Invalid transfer mechanism for {spec['wisdom_class']}")
    if spec["transfer_bucket"] not in VALID_TRANSFER_BUCKETS:
        raise ValueError(f"Invalid transfer bucket for {spec['wisdom_class']}")


def build_wisdom_class_audit(repo_root: Path):
    rows = []
    for spec in WISDOM_CLASS_SPECS:
        _validate_spec(spec)
        row = dict(spec)
        row["heuristic_reference_status"] = _reference_status(repo_root, spec["heuristic_refs"])
        row["rl_reference_status"] = _reference_status(repo_root, spec["rl_refs"])
        rl_tokens = {item["token"] for item in row["rl_reference_status"] if item["present"]}
        if spec["wisdom_class"] == "stage_post_legal_cleanup" and {"cleanup_variant_head", "cleanup_aux_loss"} <= rl_tokens:
            row["wisdom_status"] = "partially_internalized"
            row["optimized"] = "partial"
        if spec["wisdom_class"] == "selection_legality_first_ranking" and {"candidate_rank_head", "ranking_aux_loss"} <= rl_tokens:
            row["wisdom_status"] = "partially_internalized"
            row["optimized"] = "partial"
        if spec["wisdom_class"] == "case_shape_routing" and {"mode_selector_head", "mode_selector_top1"} <= rl_tokens:
            row["wisdom_status"] = "partially_internalized"
            row["optimized"] = "partial"
        if spec["wisdom_class"] == "locality_translation_clean_scoring" and {"translation_clean", "cleanup_feature_vector"} & rl_tokens:
            row["wisdom_status"] = "partially_internalized"
            row["represented"] = "partial"
            row["optimized"] = "partial"
            row["separated"] = "partial"
            row["selected"] = "partial"
            row["generalized"] = "partial"
        row["reference_health"] = "ok" if all(item["present"] for item in row["rl_reference_status"]) else "partial"
        rows.append(row)
    return rows


FACTOR_TO_WISDOM = {
    "continuation_after_good_basin": ("legality_first_selection_missing", "candidate_competition_missing", "final_candidate_selection"),
    "active_set_incompleteness": ("constraint_exposure_missing", "locality_control_missing", "legalization"),
    "phr_budget_sensitivity": ("credit_assignment_too_diffuse", "locality_control_missing", "legalization"),
    "memory_sensitivity": ("credit_assignment_too_diffuse", "case_shape_routing_missing", "basin_formation"),
    "wire_recovery_missing": ("post_legal_cleanup_bias_missing", "candidate_competition_missing", "post_legal_cleanup"),
    "local_variant_sensitivity": ("post_legal_cleanup_bias_missing", "locality_control_missing", "post_legal_cleanup"),
    "macro_refine_needed": ("macro_context_missing", "post_legal_cleanup_bias_missing", "post_legal_cleanup"),
}


def infer_heuristic_pattern(row):
    macros = int(row.get("size_macros", 0))
    std_cells = int(row.get("size_std_cells", 0))
    if std_cells >= 1500:
        return "large_scale_pruning_case"
    if macros >= 7:
        return "macro_dominated_case"
    if std_cells >= 100:
        return "dense_detail_case"
    if macros >= 3 and std_cells >= 50:
        return "spread_first_case"
    return "compact_small_case"


def infer_gap_without_counterfactual(row):
    best_overlap = float(row.get("best_overlap", 1.0))
    best_pairs = int(row.get("best_exact_overlap_pairs", 9999))
    mode_sensitivity = float(row.get("mode_sensitivity", 0.0))
    variant_sensitivity = float(row.get("variant_sensitivity", 0.0))
    winner = str(row.get("winning_refine_variant", "incumbent_hold"))
    macros = int(row.get("size_macros", 0))

    if best_overlap <= 0.15 and best_pairs <= 1:
        if winner == "incumbent_hold":
            return "post_legal_cleanup_bias_missing", "candidate_competition_missing", "post_legal_cleanup"
        if variant_sensitivity > 0.10:
            return "post_legal_cleanup_bias_missing", "locality_control_missing", "post_legal_cleanup"
        return "candidate_competition_missing", "none", "final_candidate_selection"
    if macros >= 7 and best_overlap > 0.20:
        return "macro_context_missing", "case_shape_routing_missing", "basin_formation"
    if mode_sensitivity > 0.15:
        return "case_shape_routing_missing", "candidate_competition_missing", "basin_formation"
    if best_overlap > 0.25:
        return "credit_assignment_too_diffuse", "locality_control_missing", "legalization"
    return "candidate_competition_missing", "none", "final_candidate_selection"


def build_case_gap_rows(validation_rows, analyzed_rows):
    analyzed_by_key = {
        (int(row["suite_index"]), str(row["size"]), int(row["seed"])): row for row in analyzed_rows
    }
    gap_rows = []
    for row in validation_rows:
        key = (int(row["suite_index"]), str(row["size"]), int(row["seed"]))
        diagnosis_row = analyzed_by_key.get(key)
        if diagnosis_row is not None:
            dominant = str(diagnosis_row.get("dominant_factor", ""))
            primary, secondary, stage = FACTOR_TO_WISDOM.get(
                dominant,
                ("credit_assignment_too_diffuse", "none", "unknown"),
            )
            evidence = (
                f"counterfactual dominant_factor={dominant}; overlap={float(row.get('best_overlap', 0.0)):.4f}; "
                f"pairs={int(row.get('best_exact_overlap_pairs', 0))}; mode={row.get('winning_discover_mode', 'balanced')}"
            )
        else:
            primary, secondary, stage = infer_gap_without_counterfactual(row)
            evidence = (
                f"ranking-only inference; overlap={float(row.get('best_overlap', 0.0)):.4f}; "
                f"pairs={int(row.get('best_exact_overlap_pairs', 0))}; mode_sensitivity={float(row.get('mode_sensitivity', 0.0)):.4f}; "
                f"variant_sensitivity={float(row.get('variant_sensitivity', 0.0)):.4f}"
            )
        if primary not in WISDOM_LABELS or secondary not in WISDOM_LABELS:
            raise ValueError(f"Invalid wisdom labels for case {key}: {primary}, {secondary}")
        gap_rows.append(
            {
                "suite_index": int(row["suite_index"]),
                "size": str(row["size"]),
                "size_macros": int(row.get("size_macros", 0)),
                "size_std_cells": int(row.get("size_std_cells", 0)),
                "seed": int(row["seed"]),
                "winning_heuristic_pattern": infer_heuristic_pattern(row),
                "winning_rl_pattern": (
                    f"discover:{row.get('winning_discover_mode', 'balanced')}"
                    f"+refine:{row.get('winning_refine_variant', 'incumbent_hold')}"
                ),
                "gap_stage": stage,
                "missing_wisdom_primary": primary,
                "missing_wisdom_secondary": secondary,
                "evidence": evidence,
            }
        )
    return gap_rows


def build_wisdom_gap_summary(class_rows, gap_rows):
    primary_counts = Counter(row["missing_wisdom_primary"] for row in gap_rows)
    secondary_counts = Counter(row["missing_wisdom_secondary"] for row in gap_rows if row["missing_wisdom_secondary"] != "none")
    status_counts = Counter(row["wisdom_status"] for row in class_rows)
    transfer_counts = Counter(row["transfer_bucket"] for row in class_rows)
    return {
        "primary_gap_counts": dict(sorted(primary_counts.items())),
        "secondary_gap_counts": dict(sorted(secondary_counts.items())),
        "wisdom_status_counts": dict(sorted(status_counts.items())),
        "transfer_bucket_counts": dict(sorted(transfer_counts.items())),
    }


def build_wisdom_roadmap(class_rows, gap_rows):
    primary_counts = Counter(row["missing_wisdom_primary"] for row in gap_rows)
    rows_by_class = {row["wisdom_class"]: row for row in class_rows}
    label_to_class = {
        "legality_first_selection_missing": "selection_legality_first_ranking",
        "candidate_competition_missing": "selection_candidate_competition",
        "case_shape_routing_missing": "case_shape_routing",
        "locality_control_missing": "locality_global_vs_local",
        "bounded_escape_missing": "stage_stuck_layout_escape",
        "post_legal_cleanup_bias_missing": "stage_post_legal_cleanup",
        "macro_context_missing": "case_shape_macro_context",
        "credit_assignment_too_diffuse": "stage_basin_formation",
        "constraint_exposure_missing": "case_shape_large_scale_pruning",
    }
    roadmap = []
    for label, count in primary_counts.most_common():
        class_id = label_to_class.get(label)
        if class_id is None:
            continue
        class_row = rows_by_class[class_id]
        roadmap.append(
            {
                "priority_rank": len(roadmap) + 1,
                "missing_wisdom_label": label,
                "wisdom_class": class_id,
                "group": class_row["group"],
                "cases_flagged": int(count),
                "why_it_matters": class_row["decision_it_supports"],
                "current_rl_status": class_row["wisdom_status"],
                "transfer_mechanism": class_row["transfer_mechanism"],
                "transfer_bucket": class_row["transfer_bucket"],
                "transfer_path": class_row["transfer_path"],
                "success_metric": class_row["success_metric"],
            }
        )
    return roadmap


def write_wisdom_outputs(*, repo_root: Path, diagnosis_dir: Path, validation_rows, analyzed_rows):
    class_rows = build_wisdom_class_audit(repo_root)
    gap_rows = build_case_gap_rows(validation_rows, analyzed_rows)
    summary = build_wisdom_gap_summary(class_rows, gap_rows)
    roadmap = build_wisdom_roadmap(class_rows, gap_rows)

    class_csv_rows = []
    for row in class_rows:
        class_csv_rows.append(
            {
                "wisdom_class": row["wisdom_class"],
                "group": row["group"],
                "represented": row["represented"],
                "optimized": row["optimized"],
                "separated": row["separated"],
                "selected": row["selected"],
                "generalized": row["generalized"],
                "wisdom_status": row["wisdom_status"],
                "transfer_mechanism": row["transfer_mechanism"],
                "transfer_bucket": row["transfer_bucket"],
                "reference_health": row["reference_health"],
            }
        )

    _write_csv(
        diagnosis_dir / "wisdom_class_audit.csv",
        class_csv_rows,
        list(class_csv_rows[0].keys()) if class_csv_rows else [
            "wisdom_class",
            "group",
            "represented",
            "optimized",
            "separated",
            "selected",
            "generalized",
            "wisdom_status",
            "transfer_mechanism",
            "transfer_bucket",
            "reference_health",
        ],
    )
    _write_csv(
        diagnosis_dir / "wisdom_case_gaps.csv",
        gap_rows,
        list(gap_rows[0].keys()) if gap_rows else [
            "suite_index",
            "size",
            "seed",
            "winning_heuristic_pattern",
            "winning_rl_pattern",
            "gap_stage",
            "missing_wisdom_primary",
            "missing_wisdom_secondary",
            "evidence",
        ],
    )
    (diagnosis_dir / "wisdom_class_audit.json").write_text(
        json.dumps(class_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (diagnosis_dir / "wisdom_gap_summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "roadmap": roadmap,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (diagnosis_dir / "wisdom_roadmap.json").write_text(
        json.dumps(roadmap, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "class_rows": class_rows,
        "gap_rows": gap_rows,
        "summary": summary,
        "roadmap": roadmap,
        "output_files": {
            "wisdom_class_audit_csv": str(diagnosis_dir / "wisdom_class_audit.csv"),
            "wisdom_class_audit_json": str(diagnosis_dir / "wisdom_class_audit.json"),
            "wisdom_case_gaps_csv": str(diagnosis_dir / "wisdom_case_gaps.csv"),
            "wisdom_gap_summary_json": str(diagnosis_dir / "wisdom_gap_summary.json"),
            "wisdom_roadmap_json": str(diagnosis_dir / "wisdom_roadmap.json"),
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
    result = write_wisdom_outputs(
        repo_root=repo_root,
        diagnosis_dir=diagnosis_dir,
        validation_rows=validation_rows,
        analyzed_rows=analyzed_rows,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
