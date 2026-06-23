def replace_tag(text: str, tag: str) -> str:
    if f'<{tag}><![CDATA[' in text and f']]></{tag}>' in text:
        return text
    return text.replace(f'<{tag}>', f'<{tag}><![CDATA[').replace(f'</{tag}>', f']]>\n</{tag}>').strip()

