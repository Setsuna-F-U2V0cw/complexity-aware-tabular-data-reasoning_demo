import re
import time
import traceback
from typing import Any, Dict, List, Tuple

import pandas as pd
from openai import OpenAI

try:
    import config
except ImportError:
    from . import config


class BaselineAgent:
    def __init__(self, method: str, model_name: str = None, enable_thinking: bool = False):
        self.method = method
        self.enable_thinking = enable_thinking

        self.model_configs = {
            "retrieval": config.MODEL_CONFIG["retrieval"].copy(),
        }

        if model_name:
            for key in self.model_configs:
                self.model_configs[key]["model_name"] = model_name

        self.clients = {}
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}
        self.temperature = config.TEMPERATURE
        self.timeout = 90

    def _get_client(self, model_type: str) -> Tuple[OpenAI, str]:
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
                print(f"Using model: {model} for {model_type} task (Baseline: {self.method})")

                if model_type not in self.conversation_history:
                    self.conversation_history[model_type] = []

                self.conversation_history[model_type].extend(
                    [m for m in messages if m not in self.conversation_history[model_type]]
                )

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

                if reasoning_content:
                    merged = f"<thinking>\n{reasoning_content}\n</thinking>\n\n{content}"
                else:
                    merged = content

                self.conversation_history[model_type].append({"role": "assistant", "content": merged})

                return merged, prompt_tokens, completion_tokens
            except Exception as e:
                error_msg = f"API error: {e}"
                print(error_msg)
                if retry_count < max_retries - 1:
                    retry_count += 1
                    time.sleep(2)
                else:
                    if model_type in self.conversation_history:
                        self.conversation_history[model_type].append(
                            {"role": "system", "content": f"Error: {error_msg}"}
                        )
                    return error_msg, 0, 0
        return "Max retries exceeded", 0, 0

    def parse_code(self, response: str) -> str:
        xml_pattern = r"<code[^>]*>(.*?)</code>"
        xml_match = re.search(xml_pattern, response, re.DOTALL | re.IGNORECASE)
        if xml_match:
            return xml_match.group(1).strip()

        if "```" in response:
            code_pattern = r"```(?:python)?((.|\n)*?)```"
            code_blocks = re.findall(code_pattern, response, re.DOTALL)
            if code_blocks:
                if isinstance(code_blocks[-1], (tuple, list)):
                    return code_blocks[-1][0].strip()
                if isinstance(code_blocks[-1], str):
                    return code_blocks[-1].strip()

        other_xml_patterns = [
            r"<answer[^>]*>(.*?)</answer>",
            r"<python[^>]*>(.*?)</python>",
            r"<script[^>]*>(.*?)</script>",
            r"<codeblock[^>]*>(.*?)</codeblock>",
        ]

        for pattern in other_xml_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return response.strip()

    def get_dataframe_info(self, df: pd.DataFrame, convert_to_native_types: bool = False, max_tokens: int = None) -> str:
        sample_rows = 5
        if max_tokens and len(df) > 0:
            target_chars = max_tokens * 3.5
            try:
                if convert_to_native_types:
                    one_row_str = str(df.head(1).to_dict("records"))
                else:
                    one_row_str = df.head(1).to_string()

                row_char_len = len(one_row_str)
                if row_char_len > 0:
                    estimated_rows = int(target_chars / row_char_len)
                    sample_rows = max(5, min(estimated_rows, len(df)))
            except Exception:
                sample_rows = 5

        info: List[str] = [
            "DataFrame Information:",
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
            missing_info = []
            for col, count in missing.items():
                if count > 0:
                    missing_info.append(f"{col}: {count} missing values")
            if missing_info:
                info.append("Missing Values:\n" + "\n".join(missing_info))
        return "\n\n".join(info)

    def run_pipeline(self, df: pd.DataFrame, user_query: str) -> Tuple[str, Dict[str, Any], Dict[str, List[Dict[str, str]]]]:
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
            if self.method == "single_llm":
                return self._run_single_llm(df_info, user_query, stats)
            if self.method == "single_llm_cot":
                return self._run_cot(df_info, user_query, stats)
            if self.method == "multi_agent_debate":
                return self._run_mad(df_info, user_query, stats)
            if self.method == "plan_and_solve":
                return self._run_pos(df_info, user_query, stats)
            raise ValueError(f"Unknown baseline method: {self.method}")
        except Exception as e:
            print(f"Error in baseline {self.method}: {e}")
            traceback.print_exc()
            stats["execution_output"] = f"Error: {str(e)}"
            stats["total_time"] = time.time() - stats["start_time"]
            return "", stats, self.conversation_history

    def _run_single_llm(self, df_info: str, user_query: str, stats: Dict[str, Any]):
        prompt = f"""Given the following dataframe information and a user query, provide a direct answer.

Data Information:
{df_info}

User Query:
{user_query}

Answer:"""

        messages = [{"role": "user", "content": prompt}]

        response, pr_tok, com_tok = self.gpt_chat(messages, "retrieval")

        stats["api_calls"] += 1
        stats["prompt_tokens"] += pr_tok
        stats["completion_tokens"] += com_tok
        stats["model_usage"][self.method] += 1
        stats["execution_output"] = response
        stats["total_time"] = time.time() - stats["start_time"]

        return "", stats, self.conversation_history

    def _run_cot(self, df_info: str, user_query: str, stats: Dict[str, Any]):
        prompt = f"""Given the following dataframe information and a user query, please analyze the statistical table step by step.
First count the quantities, then perform department attribution or any other necessary logical steps.

Data Information:
{df_info}

User Query:
{user_query}

Let's think step by step.
"""
        messages = [{"role": "user", "content": prompt}]

        response, pr_tok, com_tok = self.gpt_chat(messages, "retrieval")

        stats["api_calls"] += 1
        stats["prompt_tokens"] += pr_tok
        stats["completion_tokens"] += com_tok
        stats["model_usage"][self.method] += 1
        stats["execution_output"] = response
        stats["total_time"] = time.time() - stats["start_time"]

        return "", stats, self.conversation_history

    def _run_mad(self, df_info: str, user_query: str, stats: Dict[str, Any], num_agents: int = 3, num_rounds: int = 2):
        agent_responses = []

        for i in range(num_agents):
            agent_id = f"agent_{i+1}"
            prompt = f"""Given the following dataframe information and a user query, analyze the statistical table step by step.
Count quantities first, then perform department attribution or any other necessary logical reasoning.

Data Information:
{df_info}

User Query:
{user_query}

Think step by step and provide your answer.
"""
            messages = [{"role": "user", "content": prompt}]
            response, pr_tok, com_tok = self.gpt_chat(messages, "retrieval")
            stats["api_calls"] += 1
            stats["prompt_tokens"] += pr_tok
            stats["completion_tokens"] += com_tok
            stats["model_usage"][self.method] = stats["model_usage"].get(self.method, 0) + 1

            agent_responses.append(response)
            self.conversation_history[f"{self.method}_{agent_id}"] = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]

        for _round_idx in range(num_rounds):
            new_responses = []
            for i in range(num_agents):
                agent_id = f"agent_{i+1}"
                other_responses = "\n\n".join(
                    [
                        f"### Agent {j+1}'s answer:\n{resp}"
                        for j, resp in enumerate(agent_responses)
                        if j != i
                    ]
                )

                debate_prompt = f"""Here are other agents' answers to the same question:

{other_responses}

Read the other agents' answers carefully. Point out logical flaws or errors, revise your reasoning if needed, and provide an updated final answer.
If you believe you are correct, defend your position clearly. If another agent is correct, update your answer.
State your reasoning steps clearly.
"""
                messages = self.conversation_history[f"{self.method}_{agent_id}"].copy()
                messages.append({"role": "user", "content": debate_prompt})

                response, pr_tok, com_tok = self.gpt_chat(messages, "retrieval")
                stats["api_calls"] += 1
                stats["prompt_tokens"] += pr_tok
                stats["completion_tokens"] += com_tok
                stats["model_usage"][self.method] += 1

                new_responses.append(response)
                self.conversation_history[f"{self.method}_{agent_id}"].extend(
                    [
                        {"role": "user", "content": debate_prompt},
                        {"role": "assistant", "content": response},
                    ]
                )

            agent_responses = new_responses

        all_final_responses = "\n\n".join(
            [f"### Agent {j+1}'s final answer:\n{resp}" for j, resp in enumerate(agent_responses)]
        )
        judge_prompt = f"""You are a Judge Agent. Based on the dataframe information, user query, and the agents' final answers after debate, produce one correct final answer.

Data Information:
{df_info}

User Query:
{user_query}

Agents' Final Answers:
{all_final_responses}

Synthesize the viewpoints, identify errors, extract correct reasoning, and provide a final accurate answer.
Wrap the final answer in <answer> and </answer> tags, e.g. <answer>final result</answer>.
"""
        messages = [{"role": "user", "content": judge_prompt}]
        final_response, pr_tok, com_tok = self.gpt_chat(messages, "retrieval")
        stats["api_calls"] += 1
        stats["prompt_tokens"] += pr_tok
        stats["completion_tokens"] += com_tok
        stats["model_usage"][self.method] += 1

        self.conversation_history[f"{self.method}_judge"] = [
            {"role": "user", "content": judge_prompt},
            {"role": "assistant", "content": final_response},
        ]

        parsed_answer = self.parse_code(final_response)
        stats["execution_output"] = parsed_answer
        stats["total_time"] = time.time() - stats["start_time"]

        return "", stats, self.conversation_history

    def _run_pos(self, df_info: str, user_query: str, stats: Dict[str, Any], max_retries: int = 2):
        planner_prompt = f"""You are a Planner Agent. Break the complex data query into a linear natural-language execution plan for a downstream executor.
Do not answer the question directly and do not write code. Provide numbered logical steps only.

Data Information:
{df_info}

User Query:
{user_query}

Provide your execution plan:
"""
        messages = [{"role": "user", "content": planner_prompt}]
        plan_response, pr_tok, com_tok = self.gpt_chat(messages, "retrieval")
        stats["api_calls"] += 1
        stats["prompt_tokens"] += pr_tok
        stats["completion_tokens"] += com_tok
        stats["model_usage"][self.method] = stats["model_usage"].get(self.method, 0) + 1
        self.conversation_history[f"{self.method}_planner"] = [
            {"role": "user", "content": planner_prompt},
            {"role": "assistant", "content": plan_response},
        ]

        current_executor_history = []
        feedback = ""

        for attempt in range(max_retries + 1):
            if attempt == 0:
                executor_prompt = f"""You are an Executor Agent. Follow the planner's execution plan step by step using the dataframe information, simulate the reasoning in context, and produce a preliminary answer.
Write out your reasoning in detail.

Data Information:
{df_info}

User Query:
{user_query}

Execution Plan:
{plan_response}

Begin your step-by-step reasoning and answer:
"""
                current_executor_history = [{"role": "user", "content": executor_prompt}]
            else:
                executor_prompt = f"""The Reviewer Agent found issues in your previous reasoning:

{feedback}

Revise your reasoning based on this feedback and provide an updated answer with detailed steps.
"""
                current_executor_history.append({"role": "user", "content": executor_prompt})

            executor_response, pr_tok, com_tok = self.gpt_chat(current_executor_history, "retrieval")
            stats["api_calls"] += 1
            stats["prompt_tokens"] += pr_tok
            stats["completion_tokens"] += com_tok
            stats["model_usage"][self.method] += 1
            current_executor_history.append({"role": "assistant", "content": executor_response})

            self.conversation_history[f"{self.method}_executor"] = current_executor_history.copy()

            last_chance_note = ""
            if attempt == max_retries:
                last_chance_note = """

Important: This is the final review attempt. Even if you still believe the executor's reasoning is incorrect or cannot be fully verified, you must include <answer>...</answer> with the best final conclusion you can support from the dataframe information and the executor's reasoning.
"""

            reviewer_prompt = f"""You are a Reviewer Agent. Using the dataframe information below, check whether the Executor Agent's reasoning and preliminary answer correctly address the user query.

Data Information:
{df_info}

Original User Query:
{user_query}

Executor's Reasoning and Answer:
{executor_response}
{last_chance_note}
Review requirements:
1. Verify the executor's key conclusions against the dataframe information. If the logic or answer is wrong, incomplete, or off-topic, explain the issue and suggest corrections.
2. If the executor's answer is correct and complete, include a <pass> marker and wrap the final answer in <answer>...</answer>.
3. If this is not the final review attempt and the answer fails review, do not use <pass>. On the final attempt, you must output <answer>...</answer> regardless.
"""
            reviewer_messages = [{"role": "user", "content": reviewer_prompt}]
            reviewer_response, pr_tok, com_tok = self.gpt_chat(reviewer_messages, "retrieval")
            stats["api_calls"] += 1
            stats["prompt_tokens"] += pr_tok
            stats["completion_tokens"] += com_tok
            stats["model_usage"][self.method] += 1

            self.conversation_history[f"{self.method}_reviewer_attempt_{attempt+1}"] = [
                {"role": "user", "content": reviewer_prompt},
                {"role": "assistant", "content": reviewer_response},
            ]

            if "<pass>" in reviewer_response.lower() or attempt == max_retries:
                m_ans = re.search(
                    r"<answer[^>]*>(.*?)</answer>",
                    reviewer_response,
                    re.DOTALL | re.IGNORECASE,
                )
                stats["execution_output"] = m_ans.group(1).strip() if m_ans else ""
                break

            feedback = reviewer_response

        stats["total_time"] = time.time() - stats["start_time"]
        return "", stats, self.conversation_history
