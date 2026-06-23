import os
import re
import time
import json
from typing import List, Dict, Any, Tuple

import utils.xml_to_dict as utils_xml_to_dict
import utils.parse_xml as utils_parse_xml
import utils.parse_code as utils_parse_code
import utils.replace_tag as utils_replace_tag

# Configuration
try:
    import config
except ImportError:
    # If running from a different context where implicit sibling import fails
    try:
        from . import config
    except ImportError:
        # Fallback or allow it to fail if config is missing
        import src.config as config


import pandas as pd

# BERT difficulty classifier
from difficulty_classifier import DifficultyClassifier

# OpenAI-compatible client (DashScope, DeepSeek, etc.)
from openai import OpenAI

# LangGraph
try:
    from langgraph.graph import StateGraph, END
except Exception as e:
    raise ImportError



class ExcelCodeAgent:
    """
    LangGraph-based Excel multi-agent pipeline.
    Four stages: retrieval, planning, coding and debugging, final execution.
    """

    def __init__(
        self,
        k: int = 3,
        t: int = 3,
        retrieval_model: str = None,
        planning_model: str = None,
        coding_model: str = None,
        debugging_model: str = None,
        temperature: float = None,
        timeout: int = None,
    ) -> None:
        self.k = k
        self.t = t
        
        # Load model configs
        self.model_configs = config.MODEL_CONFIG
        
        # Override model names if provided in init
        if retrieval_model:
            self.model_configs["retrieval"]["model_name"] = retrieval_model
        if planning_model:
            self.model_configs["planning"]["model_name"] = planning_model
        if coding_model:
            self.model_configs["coding"]["model_name"] = coding_model
        if debugging_model:
            self.model_configs["debugging"]["model_name"] = debugging_model

        self.temperature = temperature if temperature is not None else config.TEMPERATURE
        self.timeout = timeout if timeout is not None else config.TIMEOUT

        # Clients cache to avoid recreating for same config
        self.clients = {}

        # BERT difficulty classifier
        try:
            self.difficulty_classifier = DifficultyClassifier()
            print("BERT difficulty classifier initialized successfully")
        except Exception as e:
            print(f"BERT difficulty classifier initialization failed: {e}")
            self.difficulty_classifier = None

        # Conversation history (per agent type)
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {
            "retrieval": [],
            "planning": [],
            "coding": [],
            "debugging": [],
        }


        # Build LangGraph
        self.graph = self._build_graph()

    def _get_client(self, model_type: str) -> Tuple[OpenAI, str]:
        """
        Get or create an OpenAI client for the specified model type.
        Returns the client and the model name.
        """
        cfg = self.model_configs.get(model_type, self.model_configs["retrieval"])
        
        api_key = cfg["api_key"]
        base_url = cfg["base_url"]
        model_name = cfg["model_name"]
        
        client_key = f"{base_url}_{api_key}"
        
        if client_key not in self.clients:
            self.clients[client_key] = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            
        return self.clients[client_key], model_name

    def gpt_chat(self, messages: List[Dict[str, str]], model_type: str = "retrieval") -> Tuple[str, int, int]:
        max_retries = 2
        retry_count = 0
        
        client, model = self._get_client(model_type)

        while retry_count < max_retries:
            try:
                print(f"Using model: {model} for {model_type} task")

                # Maintain per-stage conversation history
                if model_type in self.conversation_history:
                    if not self.conversation_history[model_type]:
                        self.conversation_history[model_type].extend(messages)
                    else:
                        if messages and len(messages) > 0:
                            self.conversation_history[model_type].append(messages[-1])

                print("Preparing request...")
                time.sleep(1.5)

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    max_tokens=None,  # Explicit max output tokens to avoid premature truncation or unbounded generation
                    extra_body={
                        "enable_thinking": False, # API
                    #    "chat_template_kwargs": {"enable_thinking": True}, # VLLM
                    }
                )

                content = response.choices[0].message.content or ""
                
                reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None) or ""
                
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                completion_tokens = getattr(response.usage, "completion_tokens", 0)

                if not content or not isinstance(content, str):
                    if retry_count < max_retries - 1:
                        print(f"Empty response content, retry {retry_count + 1}...")
                        retry_count += 1
                        time.sleep(3)
                        continue
                try:
                    _ = response.choices[0].message
                except Exception:
                    if retry_count < max_retries - 1:
                        print(f"Invalid response structure, retry {retry_count + 1}...")
                        retry_count += 1
                        time.sleep(3)
                        continue

                if model_type in self.conversation_history:
                    self.conversation_history[model_type].append({"role": "assistant", "content": content})

                if reasoning_content:
                    content = f"<thinking>\n{reasoning_content}\n</thinking>\n\n{content}"

                return content, prompt_tokens, completion_tokens
            except Exception as e:
                error_msg = f"API call error: {e}"
                print(error_msg)
                if retry_count < max_retries - 1:
                    print(f"Exception occurred, retry {retry_count + 1}...")
                    retry_count += 1
                    time.sleep(3)
                else:
                    if model_type in self.conversation_history:
                        self.conversation_history[model_type].append({"role": "system", "content": f"Error: {error_msg}"})
                    return error_msg, 0, 0

    def xml_to_dict(self, element) -> Dict:
        return utils_xml_to_dict(element)

    def parse_xml(self, response: str) -> Dict:
        return utils_parse_xml(response)

    def parse_code(self, response: str) -> str:
        return utils_parse_code(response)

    def replace_tag(self, text: str, tag: str) -> str:
        return utils_replace_tag(text, tag)

    def get_dataframe_info(self, df: pd.DataFrame, convert_to_native_types: bool = False, max_tokens: int = None) -> str:
        # convert_to_native_types False: human-readable; True: LLM-friendly native types
        
        sample_rows = 5
        if max_tokens and len(df) > 0:
            # Estimate: 1 token ≈ 3.5 chars (conservative)
            target_chars = max_tokens * 3.5
            try:
                if convert_to_native_types:
                    one_row_str = str(df.head(1).to_dict('records'))
                else:
                    one_row_str = df.head(1).to_string()
                
                row_char_len = len(one_row_str)
                if row_char_len > 0:
                    estimated_rows = int(target_chars / row_char_len)
                    sample_rows = max(5, min(estimated_rows, len(df)))
                    print(f"Dynamic sample row count: {sample_rows} (target tokens: {max_tokens}, ~chars per row: {row_char_len})")
            except Exception as e:
                print(f"Error estimating rows, using default 5: {e}")
                sample_rows = 5

        info: List[str] = [
            f"DataFrame Information:",
            f"Shape: {df.shape}",
            f"Columns: {list(df.columns)}",
            f"Data Types: {df.dtypes.to_dict()}",
        ]
        if convert_to_native_types:
            info.append(f"Sample Data (First {sample_rows} rows):\n{str(df.head(sample_rows).to_dict('records'))}")
        else:
            info.append(f"Sample Data (First {sample_rows} rows):\n{df.head(sample_rows).to_string()}")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            missing_info: List[str] = []
            for col, count in missing.items():
                if count > 0:
                    missing_info.append(f"{col}: {count} missing values")
            if missing_info:
                info.append("Missing Values:\n" + "\n".join(missing_info))
        return "\n\n".join(info)

    def test_code(self, df: pd.DataFrame, code: str) -> Tuple[bool, str, str]:
        local_vars: Dict[str, Any] = {"df": df.copy(), "pd": pd}
        try:
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                exec(code, globals(), local_vars)
            output = f.getvalue()
            return True, "", output
        except Exception as e:
            error_msg = f"Error type: {type(e).__name__}\nError message: {str(e)}"
            print(error_msg)
            return False, error_msg, ""

    # -------------------- LangGraph nodes (English prompts) --------------------
    def _node_retrieval(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("===== Retrieval agent phase =====")
        df_info: str = state["df_info"]
        user_query: str = state["user_query"]
        k: int = self.k

        input_kb_exemplars = [
            {
                "role": "user",
                "content": f"""Given a data-science-style question (e.g., Excel-style data processing, cleaning, etc.), based on the specific question below, provide related example problems and identify algorithms or methods suitable for solving them.

# Data information:
{df_info}
Note: Only a small subset of the data is shown above. The full table has been loaded into pandas as variable `df`.

# User query:
{user_query}

# Examples:
Recall {k} related but distinct data-processing problems (different from the question above). For each problem:
1. Describe the problem
2. Provide code using pandas and other Python libraries to solve it
3. Produce planning steps to solve it

# Algorithms / methods:

----------------
Important:
Your answer MUST follow this XML structure:

<root>
<problem>
<description>
# Problem description
</description>
<code>
# Code using pandas and other Python libraries
</code>
<planning>
# Planning steps to solve the problem
</planning>
</problem>

# Add more <problem> blocks similarly...

<algorithm>
# Identify algorithms or methods needed for the original problem (cleaning, transformation, analysis, visualization, etc.)
# Provide useful tutorials on these methods. Give a high-level generic tutorial for this class of problems. Do not generate code.
</algorithm>
</root> 
"""
            }
        ]

        content, pr_tok, com_tok = self.gpt_chat(input_kb_exemplars, "retrieval")
        content = self.replace_tag(content, 'algorithm')
        content = self.replace_tag(content, 'description')
        content = self.replace_tag(content, 'code')
        content = self.replace_tag(content, 'planning')
        retrieval_result = self.parse_xml(content)

        stats = state["stats"]
        stats["api_calls"] += 1
        stats["prompt_tokens"] += pr_tok
        stats["completion_tokens"] += com_tok
        stats["model_usage"]["retrieval"] += 1

        state.update({"retrieval_result": retrieval_result})
        print("Retrieval agent finished\n")
        return state

    def _node_planning(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("===== Planning agent phase =====")
        df_info: str = state["df_info"]
        user_query: str = state["user_query"]
        retrieval_result: Dict[str, Any] = state.get("retrieval_result", {})
        if not retrieval_result or 'problem' not in retrieval_result:
            print("Invalid retrieval result; cannot continue planning")
            state["planning_results"] = []
            return state

        target_planning_count = 3
        difficulty_label = "unknown"

        if self.difficulty_classifier:
            try:
                difficulty = self.difficulty_classifier.predict(user_query)
                if difficulty == 0:
                    difficulty_label = "easy"
                    target_planning_count = 1
                else:
                    difficulty_label = "hard"
                    target_planning_count = 3
                
                print(f"Query difficulty: {difficulty_label} ({difficulty})")
                print(f"Will generate {target_planning_count} plan(s)")
            except Exception as e:
                print(f"BERT classifier failed, using default strategy: {e}")
        else:
            print("BERT classifier unavailable; using default strategy (3 plans)")

        algorithm_prompt = f"## Related algorithms / methods:\n{retrieval_result.get('algorithm', '')}"
        plannings: List[Tuple[str, int, Dict[str, Any]]] = []
        total_pr_tok = 0
        total_com_tok = 0

        problems = retrieval_result["problem"]
        if isinstance(problems, dict):
            problems = [problems]

        selected_examples = problems[:target_planning_count] if len(problems) >= target_planning_count else problems

        for i, example in enumerate(selected_examples):
            if not isinstance(example, dict):
                continue
            if i >= target_planning_count:
                break

            example_problem = example.get("description", "")
            example_planning = example.get("planning", "")

            input_for_problem_planning = [
                {
                    "role": "user",
                    "content": f"""Given an Excel-style data processing task, produce a concrete solution plan.

# Example problem:
{example_problem}

# Example plan:
{example_planning}

{algorithm_prompt}

## Task to solve:
Data information:
{df_info}

User query:
{user_query}

## Plan:

----------------
Important:
1. Only a sample of the table is shown, but the full table is available as pandas variable `df`.
2. Provide only the planning steps to solve the problem. Do not add extra explanation or prose.
3. The plan must be for a Python script. Do NOT produce Jupyter Notebook style (no `!pip`, no `%` magics).
4. You do not need to use libraries such as matplotlib for visualization. """
                }
            ]

            planning, pr_tok, com_tok = self.gpt_chat(input_for_problem_planning, "planning")
            total_pr_tok += pr_tok
            total_com_tok += com_tok

            input_for_planning_verification = [
                {
                    "role": "user",
                    "content": f"""Given an Excel-style data processing task and a proposed solution plan, judge whether the plan is correct.

# Problem:
Data information:
{df_info}
Note: Only a small subset of the data is shown above. The full table is loaded as pandas variable `df`.

User query:
{user_query}

# Plan:
{planning}

----------------
Important: Your answer MUST follow this XML format:
<root>
<explanation>Discuss whether the plan above can solve the stated data processing task.</explanation>
<confidence>An integer from 0 to 100 representing your confidence in the plan's feasibility.</confidence>
</root> """
                }
            ]

            verification_res, pr_tok2, com_tok2 = self.gpt_chat(input_for_planning_verification, "planning")
            total_pr_tok += pr_tok2
            total_com_tok += com_tok2

            verification_res = self.replace_tag(verification_res, 'explanation')
            verification_res = self.replace_tag(verification_res, 'confidence')
            verification_dict = self.parse_xml(verification_res)
            try:
                confidence = int(str(verification_dict.get('confidence', '0')).strip())
            except Exception:
                confidence = 0

            plannings.append((planning, confidence, example))

        stats = state["stats"]
        stats["api_calls"] += len(plannings) * 2
        stats["prompt_tokens"] += total_pr_tok
        stats["completion_tokens"] += total_com_tok
        stats["model_usage"]["planning"] += len(plannings) * 2

        state["difficulty_info"] = {"difficulty": difficulty_label, "planning_count": target_planning_count}

        plannings.sort(key=lambda x: x[1], reverse=True)
        state["planning_results"] = plannings
        print(f"Planning agent finished; difficulty '{difficulty_label}', generated {len(plannings)} plan(s)\n")
        return state

    def _node_coding_debugging(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("===== Coding and debugging agent phase =====")
        df: pd.DataFrame = state["df"]
        df_info: str = state["df_info"]
        user_query: str = state["user_query"]
        planning_results: List = state.get("planning_results", [])
        if not planning_results:
            state["final_code"] = "No valid plan could be generated"
            return state

        total_coding_pr_tok = 0
        total_coding_com_tok = 0
        total_debugging_pr_tok = 0
        total_debugging_com_tok = 0

        last_code = ""
        for planning, confidence, example in planning_results:
            print(f"Trying plan (confidence: {confidence}%)...")

            input_for_code_generation = [
                {
                    "role": "user",
                    "content": f"""Given an Excel-style data processing task, generate Python code that solves it.
## Data information:
{df_info}

## Data sample:
Only a partial sample of the table is shown below. The full data is available as pandas variable `df`.

## User query:
{user_query}

## Plan:
{planning}

## Let's think step by step.

----------------
Important:
1. Only a small subset of the data is shown above; the full table is in `df`.
2. Your answer must contain ONLY Python code. No extra explanation or text.
3. Use pandas and any other necessary Python libraries.
4. Do NOT use Jupyter-only features: no magics (`%matplotlib inline`, `%pip install`, etc.) or shell lines (`!pip install`, etc.). Produce standard Python script code.
5. Do NOT use a bare variable name as the last line to display output (that is Jupyter behavior). In a script, use `print()` to show results.
6. **Do not write any file-export code** — in particular, do not use `to_csv()`, `to_excel()`, or any other file write.
7. **End by printing the data processing result only** — no saving to files.

# Good vs bad style:
Wrong (Jupyter-style):
  df.head()
  result

Correct (Python script):
  print(df.head())
  print(result) """
                }
            ]

            code, pr_tok, com_tok = self.gpt_chat(input_for_code_generation, "coding")
            total_coding_pr_tok += pr_tok
            total_coding_com_tok += com_tok
            code = self.parse_code(code)

            ok, error_msg, _ = self.test_code(df, code)
            if ok:
                print("Code test passed!")
                state["final_code"] = code
                stats = state["stats"]
                stats["api_calls"] += 1
                stats["prompt_tokens"] += pr_tok
                stats["completion_tokens"] += com_tok
                stats["model_usage"]["coding"] += 1
                return state

            last_code = code
            for i in range(1, self.t + 1):
                print(f"Debug attempt {i}/{self.t}...")
                input_for_debugging = [
                    {
                        "role": "user",
                        "content": f"""Given an Excel-style data processing task, the Python code you generated does not run correctly. Improve the code so it solves the problem.

## Data information:
{df_info}

## Data sample:
Only a partial sample of the table is shown below. The full data is available as pandas variable `df`.

## User query:
{user_query}

## Current code:
```python
{last_code}
```

## Error message:
{error_msg}

## Plan:
{planning}

## Let's think step by step about how to fix the code.

----------------
Important:
1. Only a small subset of the data is shown above; the full table is in `df`.
2. Your answer must contain ONLY the revised Python code. No extra explanation or text.
3. Use pandas and any other necessary Python libraries.
4. Do NOT use Jupyter-only features: no magics (`%matplotlib inline`, `%pip install`, etc.) or shell lines (`!pip install`, etc.). Produce standard Python script code.
5. Do NOT use a bare variable name as the last line to display output. In a script, use `print()` to show results.
6. **Do not write any file-export code** — in particular, do not use `to_csv()`, `to_excel()`, or any other file write.
7. **End by printing the data processing result only** — no saving to files.

# Good vs bad style:
Wrong (Jupyter-style):
   df.head()
   result

Correct (Python script):
   print(df.head())
   print(result) """
                    }
                ]

                improved_code, pr_tok_d, com_tok_d = self.gpt_chat(input_for_debugging, "debugging")
                total_debugging_pr_tok += pr_tok_d
                total_debugging_com_tok += com_tok_d

                improved_code = self.parse_code(improved_code)
                ok2, error_msg, _ = self.test_code(df, improved_code)
                if ok2:
                    print("Code test passed after debugging!")
                    state["final_code"] = improved_code
                    stats = state["stats"]
                    stats["api_calls"] += 1
                    stats["prompt_tokens"] += pr_tok + total_debugging_pr_tok
                    stats["completion_tokens"] += com_tok + total_debugging_com_tok
                    stats["model_usage"]["coding"] += 1
                    stats["model_usage"]["debugging"] += (1 if total_debugging_pr_tok > 0 else 0)
                    return state

                last_code = improved_code

            print("All debug attempts failed for this plan; trying the next plan...\n")

        print("No plan produced valid code; returning the last generated code")
        state["final_code"] = last_code

        stats = state["stats"]
        stats["api_calls"] += 1
        stats["model_usage"]["coding"] += 1
        return state

    def _node_final_execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n===== Final code execution =====")
        df: pd.DataFrame = state["df"]
        final_code: str = state.get("final_code", "")
        ok, err, output = self.test_code(df, final_code)
        
        state["execution_output"] = output
        
        if ok:
            print("Final code executed successfully!")
            state["final_execute_ok"] = ok
        else:
            print(f"Final code execution failed: {err}")
            state["final_execute_error"] = err

        stats = state["stats"]
        stats["total_time"] = time.time() - stats["start_time"]
        return state

    

    def _build_graph(self):
        graph = StateGraph(dict)
        graph.add_node("retrieval", self._node_retrieval)
        graph.add_node("planning", self._node_planning)
        graph.add_node("coding_debugging", self._node_coding_debugging)
        graph.add_node("final_execute", self._node_final_execute)

        graph.set_entry_point("retrieval")
        graph.add_edge("retrieval", "planning")
        graph.add_edge("planning", "coding_debugging")
        graph.add_edge("coding_debugging", "final_execute")
        graph.add_edge("final_execute", END)

        return graph.compile()

    def run_pipeline(self, df: pd.DataFrame, user_query: str) -> Tuple[str, Dict[str, Any], Dict[str, List[Dict[str, str]]]]:

        df_info = self.get_dataframe_info(df, convert_to_native_types=True, max_tokens=8192)

        stats: Dict[str, Any] = {
            "api_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "start_time": time.time(),
            "model_usage": {
                "retrieval": 0,
                "planning": 0,
                "coding": 0,
                "debugging": 0,
            },
        }

        initial_state: Dict[str, Any] = {
            "df": df,
            "df_info": df_info,
            "user_query": user_query,
            "stats": stats,
        }

        final_state = self.graph.invoke(initial_state)
        final_code = final_state.get("final_code", "")
        
        stats = final_state.get("stats", stats)
        stats["execution_output"] = final_state.get("execution_output", "")
        
        return final_code, stats, self.conversation_history
