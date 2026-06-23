import argparse
import csv
import json
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.verify import VerificationClient

PROMPTS_ROOT = os.path.join(_PROJECT_ROOT, "data", "prompts")
EVALUATED_ID_PREFIXES = ("stat", "risk")

_tls = threading.local()


def _accuracy(correct: int, total: int) -> float:
    return 0.0 if total <= 0 else correct / total * 100.0


def load_prompts(dataset_name: str, prompts_root: str) -> List[Dict[str, Any]]:
    path = os.path.join(prompts_root, f"{dataset_name}.json")
    if not os.path.exists(path):
        print(f"Warning: prompt file not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_repeat_roots(results_dir: str) -> List[Tuple[str, str]]:
    """Return [(label, path), ...] for single-run or multi-repeat layouts."""
    if not os.path.isdir(results_dir):
        return []

    def has_result_dirs(path: str) -> bool:
        return any(
            name.endswith("_results") and os.path.isdir(os.path.join(path, name))
            for name in os.listdir(path)
        )

    if has_result_dirs(results_dir):
        return [(os.path.basename(os.path.normpath(results_dir)), results_dir)]

    return sorted(
        (name, os.path.join(results_dir, name))
        for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name))
        and has_result_dirs(os.path.join(results_dir, name))
    )


def build_run_index(result_dir: str) -> Dict[str, str]:
    return {
        name[len("run_") :]: os.path.join(result_dir, name)
        for name in os.listdir(result_dir)
        if name.startswith("run_") and os.path.isdir(os.path.join(result_dir, name))
    }


def get_execution_output(run_path: str) -> str:
    stats_path = os.path.join(run_path, "stats.json")
    if not os.path.exists(stats_path):
        return ""
    with open(stats_path, "r", encoding="utf-8") as f:
        return str(json.load(f).get("execution_output", "")).strip()


def collect_jobs(
    repeat_roots: List[Tuple[str, str]],
    prompts_root: str,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str], List[int]]]:
    """Collect verification jobs and pre-record failures for missing/empty runs."""
    pending: List[Dict[str, Any]] = []
    counts: Dict[Tuple[str, str, str], List[int]] = defaultdict(lambda: [0, 0])

    for repeat_label, repeat_path in repeat_roots:
        for result_dir in sorted(
            name for name in os.listdir(repeat_path) if name.endswith("_results")
        ):
            dataset_name = result_dir.removesuffix("_results")
            full_result_dir = os.path.join(repeat_path, result_dir)
            prompts = load_prompts(dataset_name, prompts_root)
            tasks = [
                t
                for t in prompts
                if str(t.get("id", "")).startswith(EVALUATED_ID_PREFIXES)
            ]
            if not tasks:
                continue

            run_index = build_run_index(full_result_dir)
            print(
                f"Collecting {repeat_label}/{dataset_name}: "
                f"{len(tasks)} task(s), {len(run_index)} run folder(s)"
            )

            for task in tasks:
                task_id = str(task["id"])
                task_type = str(task.get("category", "unknown"))
                key = (repeat_label, dataset_name, task_type)
                run_path = run_index.get(task_id)
                output = get_execution_output(run_path) if run_path else ""

                if not output:
                    counts[key][1] += 1
                    continue

                pending.append(
                    {
                        "key": key,
                        "task_id": task_id,
                        "output": output,
                        "ground_truth": task["ground_truth"],
                    }
                )

    return pending, counts


def run_verifications(
    jobs: List[Dict[str, Any]], verify_cfg: Dict[str, str], max_workers: int
) -> List[bool]:
    if not jobs:
        return []

    def client() -> VerificationClient:
        if not hasattr(_tls, "client"):
            _tls.client = VerificationClient(verbose=False, **verify_cfg)
        return _tls.client

    def verify_one(index: int, job: Dict[str, Any]) -> Tuple[int, bool]:
        try:
            ok = client().verify(str(job["output"]), str(job["ground_truth"]))
        except Exception as e:
            print(f"Verification failed for {job['task_id']}: {e}")
            ok = False
        return index, ok

    results = [False] * len(jobs)
    print(f"Running {len(jobs)} verification call(s) with max_workers={max_workers}...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(verify_one, i, job) for i, job in enumerate(jobs)
        ]
        done = 0
        for future in as_completed(futures):
            index, ok = future.result()
            results[index] = ok
            done += 1
            if done == len(jobs) or done % 10 == 0:
                print(f"  Progress: {done}/{len(jobs)}")
    return results


def aggregate_by_type(
    counts: Dict[Tuple[str, str, str], List[int]],
) -> Dict[str, List[int]]:
    by_type: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for (_, _, task_type), (correct, total) in counts.items():
        by_type[task_type][0] += correct
        by_type[task_type][1] += total
    return by_type


def save_results(
    output_dir: str,
    counts: Dict[Tuple[str, str, str], List[int]],
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    detail_rows = []
    for (repeat_id, dataset, task_type), (correct, total) in sorted(counts.items()):
        detail_rows.append(
            {
                "repeat_id": repeat_id,
                "dataset": dataset,
                "task_type": task_type,
                "correct": correct,
                "total": total,
                "accuracy": f"{_accuracy(correct, total):.2f}%",
            }
        )

    csv_path = os.path.join(output_dir, "evaluation_by_repeat_dataset_type.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "repeat_id",
                "dataset",
                "task_type",
                "correct",
                "total",
                "accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"Saved detail CSV to {csv_path}")

    by_type = aggregate_by_type(counts)
    global_correct = sum(v[0] for v in counts.values())
    global_total = sum(v[1] for v in counts.values())

    summary_lines = ["=== Summary by task type ===\n"]
    for task_type in sorted(by_type):
        correct, total = by_type[task_type]
        acc = _accuracy(correct, total)
        summary_lines.append(f"{task_type}: {correct}/{total} = {acc:.2f}%\n")
        print(f"{task_type}: {correct}/{total} = {acc:.2f}%")

    overall_acc = _accuracy(global_correct, global_total)
    summary_lines.append(
        f"\nOverall: {global_correct}/{global_total} = {overall_acc:.2f}%\n"
    )
    print(f"\nOverall: {global_correct}/{global_total} = {overall_acc:.2f}%")

    summary_path = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(summary_lines)
    print(f"Saved summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate stat/risk experiment results with LLM verification."
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Experiment results root (single run or parent of repeat subdirs)",
    )
    parser.add_argument(
        "--prompts_root",
        default=PROMPTS_ROOT,
        help="Directory containing dataset prompt JSON files",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory; defaults to --results_dir",
    )
    parser.add_argument("--verify_model", default="qwen3-4b")
    parser.add_argument(
        "--verify_base_url",
        default=os.getenv("VERIFY_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--verify_api_key",
        default=os.getenv("VERIFY_API_KEY", "EMPTY"),
    )
    parser.add_argument("--max_workers", type=int, default=4)
    args = parser.parse_args()

    results_root = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(args.output_dir or results_root)
    verify_cfg = {
        "model": args.verify_model,
        "base_url": args.verify_base_url,
        "api_key": args.verify_api_key,
    }

    print(f"Results root: {results_root}")
    print(f"Prompts root: {args.prompts_root}")
    print(f"Verify model: {args.verify_model}")

    if not os.path.isdir(results_root):
        raise SystemExit(f"Error: results directory does not exist: {results_root}")

    repeat_roots = discover_repeat_roots(results_root)
    if not repeat_roots:
        raise SystemExit(
            f"Error: no evaluable layout under {results_root}. "
            "Expected *_results subdirs directly or under repeat subdirs."
        )

    pending, counts = collect_jobs(repeat_roots, args.prompts_root)
    if pending:
        for job, ok in zip(pending, run_verifications(pending, verify_cfg, args.max_workers)):
            key = job["key"]
            counts[key][1] += 1
            if ok:
                counts[key][0] += 1
    elif not counts:
        raise SystemExit("No evaluable tasks found.")

    save_results(output_dir, counts)


if __name__ == "__main__":
    main()
