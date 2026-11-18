import time
import traceback
from typing import Any, Dict, List, Tuple

import pandas as pd

try:
    from baselines import BaselineAgent
except ImportError:
    from .baselines import BaselineAgent


class SingleCodeAgent(BaselineAgent):
    """Single-code-generation Agent baseline with an optional debug loop."""

    def __init__(
        self,
        method: str,
        model_name: str = None,
        max_debug_attempts: int = 3,
        enable_thinking: bool = False,
    ):
        super().__init__(method=method, model_name=model_name, enable_thinking=enable_thinking)
        self.max_debug_attempts = max_debug_attempts

    def gpt_chat(self, messages: List[Dict[str, str]], model_type: str = "retrieval") -> Tuple[str, int, int]:
        max_retries = 2
        retry_count = 0

        client, model = self._get_client(model_type)

        while retry_count < max_retries:
            try:
                print(f"Using model: {model} for {model_type} task (Baseline: {self.method})")

                # Record the whole single-agent conversation under one key.
                history = self.conversation_history.setdefault(self.method, [])
                if not history:
                    history.extend(messages)
                elif messages:
                    history.append(messages[-1])

                time.sleep(1.5)

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    max_tokens=None,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
                    },
                )

                content = response.choices[0].message.content or ""
                reasoning_content = getattr(response.choices[0].message, "reasoning_content", None) or ""

                prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0

                if not content or not isinstance(content, str):
                    if retry_count < max_retries - 1:
                        print("Empty response, retrying...")
                        retry_count += 1
                        time.sleep(2)
                        continue
                try:
                    _ = response.choices[0].message
                except Exception:
                    if retry_count < max_retries - 1:
                        print("Invalid response structure, retrying...")
                        retry_count += 1
                        time.sleep(2)
                        continue

                # Keep the raw content in history; thinking is only merged into the return value.
                history = self.conversation_history.setdefault(self.method, [])
                history.append({"role": "assistant", "content": content})

                if reasoning_content:
                    content = f"<thinking>\n{reasoning_content}\n</thinking>\n\n{content}"

                return content, prompt_tokens, completion_tokens
            except Exception as e:
                error_msg = f"API error: {e}"
                print(error_msg)
                if retry_count < max_retries - 1:
                    retry_count += 1
                    time.sleep(2)
                else:
                    history = self.conversation_history.get(self.method)
                    if history is not None:
                        history.append({"role": "system", "content": f"Error: {error_msg}"})
                    return error_msg, 0, 0

        return "Max retries exceeded", 0, 0

    def test_code(self, df: pd.DataFrame, code: str) -> Tuple[bool, str, str]:
        # Execute generated code in an isolated local namespace, consistent with ExcelCodeAgent.
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

    def _build_generation_prompt(self, df_info: str, user_query: str) -> str:
        return f"""Given an Excel-style data processing task, generate Python code that solves it.
## Data information:
{df_info}

## Data sample:
Only a partial sample of the table is shown below. The full data is available as pandas variable `df`.

## User query:
{user_query}

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
  print(result)"""

    def _build_debug_prompt(
        self,
        df_info: str,
        user_query: str,
        code: str,
        error_msg: str,
    ) -> str:
        return f"""Given an Excel-style data processing task, the Python code you generated does not run correctly. Improve the code so it solves the problem.

## Data information:
{df_info}

## Data sample:
Only a partial sample of the table is shown below. The full data is available as pandas variable `df`.

## User query:
{user_query}

## Current code:
```python
{code}
```

## Error message:
{error_msg}

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
   print(result)"""

    def _run_code_agent(self, df: pd.DataFrame, df_info: str, user_query: str, stats: Dict[str, Any]):
        messages = [{"role": "user", "content": self._build_generation_prompt(df_info, user_query)}]

        code_response, pr_tok, com_tok = self.gpt_chat(messages, "coding")
        code = self.parse_code(code_response)
        ok, error_msg, output = self.test_code(df, code)

        # Match ExcelCodeAgent accounting: one api call per code plan, and token
        # usage is only recorded when code generation succeeds.
        stats["api_calls"] += 1
        stats["model_usage"][self.method] += 1
        if ok:
            stats["prompt_tokens"] += pr_tok
            stats["completion_tokens"] += com_tok

        stats["execution_output"] = output if ok else f"Code Execution Failed: {error_msg}"
        stats["total_time"] = time.time() - stats["start_time"]
        return code, stats, self.conversation_history

    def _run_code_agent_debug(self, df: pd.DataFrame, df_info: str, user_query: str, stats: Dict[str, Any]):
        messages = [{"role": "user", "content": self._build_generation_prompt(df_info, user_query)}]

        code_response, pr_tok, com_tok = self.gpt_chat(messages, "coding")
        code = self.parse_code(code_response)
        ok, error_msg, output = self.test_code(df, code)

        if ok:
            stats["api_calls"] += 1
            stats["model_usage"][self.method] += 1
            stats["prompt_tokens"] += pr_tok
            stats["completion_tokens"] += com_tok
        else:
            debugging_prompt_tokens = 0
            debugging_completion_tokens = 0
            debug_succeeded = False

            for _ in range(self.max_debug_attempts):
                debug_prompt = self._build_debug_prompt(df_info, user_query, code, error_msg)
                debug_response, pr_tok_d, com_tok_d = self.gpt_chat(
                    [{"role": "user", "content": debug_prompt}],
                    "coding",
                )
                debugging_prompt_tokens += pr_tok_d
                debugging_completion_tokens += com_tok_d

                code = self.parse_code(debug_response)
                ok, error_msg, output = self.test_code(df, code)
                if ok:
                    debug_succeeded = True
                    break

            # Match ExcelCodeAgent accounting: one api call per code plan.
            stats["api_calls"] += 1
            stats["model_usage"][self.method] += 1
            if debug_succeeded:
                stats["prompt_tokens"] += pr_tok + debugging_prompt_tokens
                stats["completion_tokens"] += com_tok + debugging_completion_tokens

        stats["execution_output"] = output if ok else f"Code Execution Failed: {error_msg}"
        stats["total_time"] = time.time() - stats["start_time"]
        return code, stats, self.conversation_history

    def run_pipeline(
        self,
        df: pd.DataFrame,
        user_query: str,
    ) -> Tuple[str, Dict[str, Any], Dict[str, List[Dict[str, str]]]]:
        df_info = self.get_dataframe_info(df, convert_to_native_types=True, max_tokens=8192)

        stats: Dict[str, Any] = {
            "api_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "start_time": time.time(),
            "model_usage": {self.method: 0},
            "execution_output": "",
        }

        self.conversation_history = {self.method: []}

        try:
            if self.method == "single_llm_code":
                return self._run_code_agent(df, df_info, user_query, stats)
            elif self.method == "single_llm_code_debug":
                return self._run_code_agent_debug(df, df_info, user_query, stats)
            else:
                raise ValueError(f"Unknown baseline method: {self.method}")
        except Exception as e:
            print(f"Error in baseline {self.method}: {e}")
            traceback.print_exc()
            stats["execution_output"] = f"Error: {str(e)}"
            stats["total_time"] = time.time() - stats["start_time"]
            return "", stats, self.conversation_history
