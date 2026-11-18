import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from baselines import BaselineAgent
from baselines_single_code import SingleCodeAgent
from excel_agent import ExcelCodeAgent, config

DEFAULT_DATASETS = [
    "data_example",
]

BASELINE_METHODS = [
    "single_llm",
    "single_llm_cot",
    "multi_agent_debate",
    "plan_and_solve",
    "single_llm_code",
    "single_llm_code_debug",
]

SINGLE_CODE_METHODS = {
    "single_llm_code",
    "single_llm_code_debug",
}


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dataset(dataset_id: str):
    project_root = _project_root()
    data_path = os.path.join(project_root, "data", f"{dataset_id}.csv")
    prompt_path = os.path.join(project_root, "data", "prompts", f"{dataset_id}.json")

    if not os.path.exists(data_path) or not os.path.exists(prompt_path):
        print(f"Missing data or prompt for {dataset_id}")
        return None, None, None, None

    try:
        df = pd.read_csv(data_path, on_bad_lines="warn", engine="python")
    except Exception as e:
        print(f"Error reading csv for {dataset_id}: {e}")
        return None, None, None, None

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
    except Exception as e:
        print(f"Error reading prompts for {dataset_id}: {e}")
        return None, None, None, None

    return project_root, df, test_cases, dataset_id


def _save_run_result(results_root_dir: str, question_id: str, final_code, stats, conversation_history):
    run_dir = os.path.join(results_root_dir, f"run_{question_id}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "final_code.py"), "w", encoding="utf-8") as f:
        f.write(final_code or "")

    with open(os.path.join(run_dir, "conversation_history.json"), "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=2)

    with open(os.path.join(run_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def run_proposed_method(dataset_id: str, run_number=None, enable_thinking: bool = False):
    project_root, df, test_cases, dataset_name = _load_dataset(dataset_id)
    if df is None:
        return

    model_name = config.MODEL_CONFIG["retrieval"]["model_name"]
    if run_number is not None:
        results_root_dir = os.path.join(
            project_root, "results", f"{model_name}-{run_number}", f"{dataset_name}_results"
        )
    else:
        results_root_dir = os.path.join(project_root, "results", model_name, f"{dataset_name}_results")

    os.makedirs(results_root_dir, exist_ok=True)
    print(f"\n{'#' * 60}")
    print(f"Processing {dataset_id} with proposed_method...")
    if run_number is not None:
        print(f"Run {run_number}")
    print(f"Results will be saved to: {results_root_dir}")
    print(f"{'#' * 60}\n")

    agent = ExcelCodeAgent(enable_thinking=enable_thinking)

    for i, item in enumerate(test_cases):
        user_query = item.get("question", "").strip()
        ground_truth = str(item.get("ground_truth", "")).strip()
        question_id = item.get("id", f"case_{i}")

        print(f"========== [{dataset_id}] Test case {i + 1} / {len(test_cases)} (ID: {question_id}) ==========")
        print(f"Question: {user_query}")

        agent.conversation_history = {
            "retrieval": [],
            "planning": [],
            "coding": [],
            "debugging": [],
        }

        try:
            final_code, stats, conversation_history = agent.run_pipeline(df, user_query)
            print(f"Output: {stats}")
            print(f"Ground truth: {ground_truth}")
            _save_run_result(results_root_dir, question_id, final_code, stats, conversation_history)
        except Exception as e:
            print(f"Test case {i + 1} failed: {e}")
            traceback.print_exc()

        print("==============================================\n")


def run_baseline(
    dataset_id: str,
    method: str,
    run_number=None,
    max_debug_attempts: int = 3,
    enable_thinking: bool = False,
):
    project_root, df, test_cases, _ = _load_dataset(dataset_id)
    if df is None:
        return

    if run_number is not None:
        results_root_dir = os.path.join(
            project_root, "results", "baseline", f"{method}-{run_number}", f"{dataset_id}_results"
        )
    else:
        results_root_dir = os.path.join(project_root, "results", "baseline", method, f"{dataset_id}_results")

    os.makedirs(results_root_dir, exist_ok=True)
    print(f"\n{'#' * 60}")
    print(f"Processing {dataset_id} with baseline method {method}...")
    if run_number is not None:
        print(f"Run {run_number}")
    print(f"Results will be saved to: {results_root_dir}")
    print(f"{'#' * 60}\n")

    if method in SINGLE_CODE_METHODS:
        agent = SingleCodeAgent(
            method=method,
            max_debug_attempts=max_debug_attempts,
            enable_thinking=enable_thinking,
        )
    else:
        agent = BaselineAgent(method=method, enable_thinking=enable_thinking)

    for i, item in enumerate(test_cases):
        user_query = item.get("question", "").strip()
        question_id = item.get("id", f"case_{i}")

        print(f"  [{dataset_id}] Test case {i + 1}/{len(test_cases)} (ID: {question_id})")

        try:
            final_code, stats, conversation_history = agent.run_pipeline(df, user_query)
            _save_run_result(results_root_dir, question_id, final_code, stats, conversation_history)
        except Exception as e:
            print(f"  [{dataset_id}] Error: {e}")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Run proposed method or baseline experiments")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["proposed_method", "baseline"],
        required=True,
        help="proposed_method: multi-agent pipeline; baseline: baseline methods",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=BASELINE_METHODS,
        help="Required when --mode baseline",
    )
    parser.add_argument("--dataset", type=str, help="Specific dataset to run (optional)")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repeated runs (default: 1)")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="Maximum concurrent datasets per run (default: 2)",
    )
    parser.add_argument(
        "--max_debug_attempts",
        type=int,
        default=3,
        help="Maximum debug attempts for single_llm_code_debug (default: 3)",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode for all gpt_chat-based agents (default: disabled)",
    )
    args = parser.parse_args()

    if args.mode == "baseline" and not args.method:
        parser.error("--method is required when --mode baseline")

    datasets = list(DEFAULT_DATASETS)
    if args.dataset:
        if args.dataset not in datasets:
            print(f"Dataset {args.dataset} not in standard list, but trying anyway.")
        datasets = [args.dataset]

    max_workers = max(1, args.max_workers)

    for run_num in range(1, args.num_runs + 1):
        print(f"\n{'#' * 60}")
        print(f"# Run {run_num} / {args.num_runs}")
        print(f"{'#' * 60}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if args.mode == "proposed_method":
                futures = {
                    executor.submit(
                        run_proposed_method,
                        dataset_id,
                        run_number=run_num,
                        enable_thinking=args.enable_thinking,
                    ): dataset_id
                    for dataset_id in datasets
                }
            else:
                futures = {
                    executor.submit(
                        run_baseline,
                        dataset_id,
                        args.method,
                        run_number=run_num,
                        max_debug_attempts=args.max_debug_attempts,
                        enable_thinking=args.enable_thinking,
                    ): dataset_id
                    for dataset_id in datasets
                }

            for future in as_completed(futures):
                dataset_id = futures[future]
                try:
                    future.result()
                    print(f"[Done] Dataset {dataset_id} completed")
                except Exception as e:
                    print(f"[Error] Dataset {dataset_id} failed: {e}")
                    traceback.print_exc()


if __name__ == "__main__":
    main()
