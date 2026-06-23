import re
import xml.etree.ElementTree as ElemTree
from typing import Any, Dict, List, Optional

from .xml_to_dict import xml_to_dict

_THINKING_PATTERN = re.compile(
    r"<think>.*?</think>", re.DOTALL
)
_MARKDOWN_FENCE_PATTERN = re.compile(r"```(?:xml)?\s*|\s*```")
_PROBLEM_FIELDS = ("description", "code", "planning")


def parse_xml(response: str) -> Dict[str, Any]:
    response = _preprocess(response)

    result = _parse_with_elementtree(response)
    if result is not None:
        return result

    print("XML parsing failed, falling back to regex extraction...")
    return _parse_with_regex(response)


def _preprocess(response: str) -> str:
    # Remove <think> blocks from reasoning model output
    response = _THINKING_PATTERN.sub("", response).strip()
    # Strip markdown code fence markers
    response = _MARKDOWN_FENCE_PATTERN.sub("", response).strip()
    return response


def _parse_with_elementtree(response: str) -> Optional[Dict[str, Any]]:
    candidates = (
        response,
        f"<root>\n{response}\n</root>",
        f"<root>\n{response}",
    )
    for candidate in candidates:
        try:
            root = ElemTree.fromstring(candidate)
            return _unwrap_root(xml_to_dict(root))
        except ElemTree.ParseError:
            continue
    return None


def _unwrap_root(result: Dict[str, Any]) -> Dict[str, Any]:
    # Unwrap nested <root> wrappers from repaired or malformed XML
    while list(result) == ["root"] and isinstance(result["root"], dict):
        result = result["root"]
    return result


def _parse_with_regex(response: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    algo_match = re.search(
        r"<algorithm>(.*?)(?:</algorithm>|$)", response, re.DOTALL
    )
    if algo_match:
        result["algorithm"] = _clean_cdata(algo_match.group(1))

    problems = _extract_problems_regex(response)
    if problems:
        result["problem"] = problems[0] if len(problems) == 1 else problems

    if not result:
        print("Regex extraction also failed, returning empty dict")
    return result


def _extract_problems_regex(response: str) -> List[Dict[str, str]]:
    problems: List[Dict[str, str]] = []
    for part in response.split("<problem>")[1:]:
        if "<algorithm>" in part:
            part = part.split("<algorithm>", 1)[0]
        if "</problem>" in part:
            part = part.split("</problem>", 1)[0]

        problem_data: Dict[str, str] = {}
        for field in _PROBLEM_FIELDS:
            match = re.search(
                rf"<{field}>(.*?)</{field}>", part, re.DOTALL
            )
            if match:
                problem_data[field] = _clean_cdata(match.group(1))

        if problem_data:
            problems.append(problem_data)
    return problems


def _clean_cdata(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if text.startswith("<![CDATA["):
        text = text[9:]
    if text.endswith("]]>"):
        text = text[:-3]
    return text.strip()
