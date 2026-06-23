import re


def parse_code(response: str) -> str:
    if "```" not in response:
        return response
    code_pattern = r'```(?:python)?((.|\n)*?)```'
    code_blocks = re.findall(code_pattern, response, re.DOTALL)
    if not code_blocks:
        return response
    if isinstance(code_blocks[-1], (tuple, list)):
        code_str = code_blocks[-1][0]
    elif isinstance(code_blocks[-1], str):
        code_str = code_blocks[-1]
    else:
        code_str = response
    return code_str.strip()


