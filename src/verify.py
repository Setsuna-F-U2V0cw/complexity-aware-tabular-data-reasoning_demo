from openai import OpenAI


class VerificationClient:
    def __init__(self, base_url=None, api_key=None, model=None, verbose=True):
        self.base_url = base_url if base_url else "http://localhost:8000/v1"
        self.api_key = api_key if api_key else "EMPTY"
        self.model = model if model else "qwen3-4b"
        self.verbose = verbose

        if self.verbose:
            print(
                f"Initialized verification client: "
                f"Base URL={self.base_url}, Model={self.model}"
            )

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def verify(self, execution_output: str, ground_truth: str) -> bool:
        """Check whether execution output and ground truth are semantically consistent."""
        prompt = f"""Please determine whether the following two results are semantically consistent.

Result 1 (Model Execution Output):
{execution_output}

Result 2 (Ground Truth):
{ground_truth}

Please ignore format differences (such as whitespace, newlines, punctuation) and focus on whether the numerical values, key information, and core semantics are the same.
If they are consistent, please output "True".
If they are not consistent, please output "False".
If the content in ```Result 1``` is a table output, you need to compare the table content. If the table content is consistent, output True; otherwise, output False.
Only output True or False, do not output any other content.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a rigorous evaluation assistant responsible for comparing whether two results are semantically consistent."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.01,
                max_tokens=None,
                stream=False,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )

            msg = response.choices[0].message
            content = (msg.content or "").strip()
            reasoning = getattr(msg, "reasoning_content", None) or ""
            if self.verbose:
                if reasoning:
                    preview = reasoning.strip()
                    if len(preview) > 500:
                        preview = preview[:500] + "…"
                    print(f"[Verify] LLM reasoning (excluded from scoring): {preview}")
                print(f"[Verify] LLM final reply (used for scoring): {content}")

            return "true" in content.lower()

        except Exception as e:
            if self.verbose:
                print(f"[Verify] Verification error: {e}")
            return False
