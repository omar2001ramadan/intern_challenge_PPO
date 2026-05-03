"""Multi-GPU inference sweep for the policy-conditioned placement challenge.

This runner evaluates a saved policy checkpoint across the benchmark cases using
declared inference controls. It no longer relies on the legacy coordinate-stage
environment variables that the current policy-conditioned path does not read.
"""

import argparse
import json
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path


TEST_CASES = [
    (1, 2, 20, 1001),
    (2, 3, 25, 1002),
    (3, 2, 30, 1003),
    (4, 3, 50, 1004),
    (5, 4, 75, 1005),
    (6, 5, 100, 1006),
    (7, 5, 150, 1007),
    (8, 7, 150, 1008),
    (9, 8, 200, 1009),
    (10, 10, 2000, 1010),
]


PROFILES = {
    "quick": {
        "PLACEMENT_POLICY_TRANSITION": "1",
        "PLACEMENT_INFERENCE_MODE": "audited_policy_ensemble",
        "PLACEMENT_POLICY_ROLLOUTS": "1",
        "PLACEMENT_POLICY_STEPS": "16",
        "PLACEMENT_POLICY_TEMPERATURE": "0.45",
    },
    "balanced": {
        "PLACEMENT_POLICY_TRANSITION": "1",
        "PLACEMENT_INFERENCE_MODE": "audited_policy_ensemble",
        "PLACEMENT_POLICY_ROLLOUTS": "2",
        "PLACEMENT_POLICY_STEPS": "24",
        "PLACEMENT_POLICY_TEMPERATURE": "0.35",
    },
    "sharp": {
        "PLACEMENT_POLICY_TRANSITION": "1",
        "PLACEMENT_INFERENCE_MODE": "audited_policy_ensemble",
        "PLACEMENT_POLICY_ROLLOUTS": "4",
        "PLACEMENT_POLICY_STEPS": "32",
        "PLACEMENT_POLICY_TEMPERATURE": "0.25",
    },
}


def parse_cases(value):
    if value == "first10":
        return TEST_CASES
    selected = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        elif part:
            selected.add(int(part))
    return [case for case in TEST_CASES if case[0] in selected]


def worker(gpu_id, jobs, results, log_path, policy_checkpoint):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PLACEMENT_DEVICE"] = "cuda:0"
    os.environ["PLACEMENT_POLICY_CHECKPOINT"] = policy_checkpoint

    from test import run_placement_test

    while True:
        try:
            profile_name, profile_env, case = jobs.get_nowait()
        except queue.Empty:
            return

        for key, value in profile_env.items():
            os.environ[key] = value

        started = time.time()
        test_id, num_macros, num_std_cells, seed = case
        try:
            result = run_placement_test(test_id, num_macros, num_std_cells, seed)
            result.update(
                {
                    "profile": profile_name,
                    "gpu": gpu_id,
                    "status": "ok",
                    "wall_time": time.time() - started,
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep long sweeps alive.
            result = {
                "profile": profile_name,
                "gpu": gpu_id,
                "test_id": test_id,
                "status": "error",
                "error": repr(exc),
                "wall_time": time.time() - started,
            }

        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        results.put(result)


def summarize(results):
    by_profile = {}
    for result in results:
        if result.get("status") != "ok":
            continue
        bucket = by_profile.setdefault(result["profile"], [])
        bucket.append(result)

    summary = {}
    for profile, rows in by_profile.items():
        summary[profile] = {
            "cases": len(rows),
            "avg_overlap": sum(row["overlap_ratio"] for row in rows) / len(rows),
            "avg_wirelength": sum(row["normalized_wl"] for row in rows) / len(rows),
            "total_runtime": sum(row["elapsed_time"] for row in rows),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--profiles", default="quick,balanced,sharp")
    parser.add_argument("--cases", default="first10")
    parser.add_argument("--log-dir", default="training_logs")
    parser.add_argument("--policy-checkpoint", default=os.environ.get("PLACEMENT_POLICY_CHECKPOINT", ""))
    args = parser.parse_args()

    if not args.policy_checkpoint:
        raise ValueError("--policy-checkpoint is required for the policy-conditioned sweep.")

    gpu_ids = [int(item) for item in args.gpus.split(",") if item.strip()]
    profile_names = [item.strip() for item in args.profiles.split(",") if item.strip()]
    cases = parse_cases(args.cases)

    unknown = [name for name in profile_names if name not in PROFILES]
    if unknown:
        raise ValueError(f"Unknown profile(s): {unknown}")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sweep_{int(time.time())}.jsonl"

    ctx = mp.get_context("spawn")
    jobs = ctx.Queue()
    results = ctx.Queue()

    for profile_name in profile_names:
        for case in cases:
            jobs.put((profile_name, PROFILES[profile_name], case))

    workers = [
        ctx.Process(target=worker, args=(gpu_id, jobs, results, str(log_path), args.policy_checkpoint))
        for gpu_id in gpu_ids
    ]

    for proc in workers:
        proc.start()

    collected = []
    remaining = len(profile_names) * len(cases)
    while remaining:
        result = results.get()
        collected.append(result)
        remaining -= 1
        print(json.dumps(result, sort_keys=True), flush=True)

    for proc in workers:
        proc.join()

    summary = summarize(collected)
    summary_path = log_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"log": str(log_path), "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
