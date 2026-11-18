# A Complexity-Aware Large Language Model Agentic Framework for Tabular Data Reasoning in Shipbuilding Inspection

Official code repository for **A Complexity-Aware Large Language Model Agentic Framework for Tabular Data Reasoning in Shipbuilding Inspection**.

This repository provides sample datasets, evaluation prompts, and a runnable demo for natural language reasoning over industrial tabular data.

## Setup

```bash
pip install -r requirements.txt
export DASHSCOPE_API_KEY="your-api-key"   # or any OpenAI-compatible API key
```

Configure LLM endpoints and model names in `src/config.py` before running experiments.

## Sample Data

The demo dataset `data_example` consists of two companion files under `data/`:

### `data/data_example.csv`

Tabular dataset example of synthetic **shipbuilding quality-inspection** records. Each row is one reported defect or non-conformance found during hull/outfitting production. Columns include:

| Column | Description |
|--------|-------------|
| `Record ID` | Unique record identifier |
| `Issue Description` | Textual description of the defect or issue |
| `Department` | Responsible department |
| `Team` | Responsible team within the department |
| `Contractor` | External contractor involved |
| `Corrective Action` | Prescribed remediation |
| `Assessment & Recommendation` | Quality assessment outcome (e.g. re-education, fine, penalty) |
| `Issue Category` | Issue type (e.g. Welding, Cutting, Assembly) |
| `Report Date` | Date the issue was reported |

### `data/prompts/data_example.json`

JSON file with **20 question–answer pairs** for evaluating tabular reasoning over the inspection log above. Each entry has:

| Field | Description |
|-------|-------------|
| `id` | Question ID (e.g. `stat_001`, `risk_003`) |
| `category` | Task type — see below |
| `question` | Natural-language question posed to the agent |
| `ground_truth` | Expected answer for evaluation |
| `evidence_rows` | Row IDs in the CSV that support the answer |

The `{dataset_id}.csv` and `prompts/{dataset_id}.json` naming convention is used throughout the codebase. For the demo, both share the prefix `data_example`.

## Running Experiments

All experiment commands should be run from the `src/` directory:

```bash
cd src
```

Results are written under `results/` at the project root. Each run produces a `{dataset_id}_results/` subdirectory with per-question outputs (`final_code.py`, `stats.json`, `conversation_history.json`).

### Proposed Method (Complexity-Aware Multi-Agent Pipeline)

```bash
python run_experiment.py --mode proposed_method
```

With multiple repeated runs (results saved to `results/{model_name}-{run_number}/`):

```bash
python run_experiment.py --mode proposed_method --num_runs 3
```

### Baseline Methods

Available baselines: `single_llm`, `single_llm_cot`, `multi_agent_debate`, `plan_and_solve`, `single_llm_code`, `single_llm_code_debug`.

```bash
python run_experiment.py --mode baseline --method multi_agent_debate
```

Single coding agent baselines generate Python code directly. Use `single_llm_code_debug` to enable the debug loop:

```bash
python run_experiment.py --mode baseline --method single_llm_code
python run_experiment.py --mode baseline --method single_llm_code_debug --max_debug_attempts 3
```

With multiple repeated runs (results saved to `results/baseline/{method}-{run_number}/`):

```bash
python run_experiment.py --mode baseline --method single_llm_cot --num_runs 3
```

Thinking mode is disabled by default. Add `--enable_thinking` to enable it for the proposed method and all `gpt_chat`-based baselines:

```bash
python run_experiment.py --mode proposed_method --enable_thinking
python run_experiment.py --mode baseline --method single_llm_code_debug --enable_thinking
```

## Evaluation

After experiments finish, evaluate results with LLM-based answer verification. Run from the **project root**:

```bash
python evaluate_results_script.py --results_dir results/{model-name}
```

For baseline results:

```bash
python evaluate_results_script.py --results_dir results/baseline/{model-name}
```

The script evaluates questions whose IDs start with `stat` or `risk`, compares agent execution output against `ground_truth` via an LLM verifier, and writes:

- `evaluation_by_repeat_dataset_type.csv` — per-category accuracy breakdown
- `evaluation_summary.txt` — overall summary

### Evaluation Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--results_dir` | *(required)* | Experiment results root directory |
| `--prompts_root` | `data/prompts` | Directory containing prompt JSON files |
| `--output_dir` | same as `--results_dir` | Where to write evaluation outputs |
| `--verify_model` | `qwen3-4b` | LLM model used for answer verification |
| `--verify_base_url` | `http://127.0.0.1:8000/v1` | OpenAI-compatible API endpoint for the verifier |
| `--verify_api_key` | `EMPTY` | API key for the verifier (or set `VERIFY_API_KEY`) |
| `--max_workers` | `4` | Concurrent verification requests |

Example with a custom verifier endpoint:

```bash
python evaluate_results_script.py \
  --results_dir results/qwen3-flash \
  --verify_model qwen3-4b \
  --verify_base_url http://127.0.0.1:8000/v1 \
  --verify_api_key EMPTY \
  --max_workers 4
```

## Embedding Model (Complexity Classifier)

The proposed method uses a fine-tuned **BGE-M3** (`BAAI/bge-m3`) classifier to estimate query complexity and route inspection questions accordingly. **Pre-trained model weights are not included in this repository.**

Download or prepare the fine-tuned classifier weights and place them under:

```
embedding_router/embedding_model/
```

The directory should contain at minimum:

```
embedding_router/embedding_model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

The classifier is loaded automatically by `src/difficulty_classifier.py` at runtime. You may fine-tune your own classifier from the base [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) checkpoint using the scripts under `embedding_router/` (not included in the open-source release).
