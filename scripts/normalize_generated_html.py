"""Strictly unwrap one permitted Markdown HTML fence and validate one document."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re


_FENCE = re.compile(r"\A```html[ \t]*\r?\n(.*?)\r?\n```[ \t]*\r?\n?\Z", re.DOTALL)
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class _StrictDocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.doctype = False
        self.root_closed = False
        self.tags: dict[str, int] = {}

    def handle_decl(self, decl: str) -> None:
        if self.doctype or self.stack or decl.strip().lower() != "doctype html":
            raise ValueError("html_declaration_invalid")
        self.doctype = True

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if not self.doctype or self.root_closed:
            raise ValueError("html_wrapper_invalid")
        if tag == "html":
            if self.tags.get("html") or self.stack:
                raise ValueError("html_root_invalid")
        elif not self.stack:
            raise ValueError("html_content_outside_root")
        names = [name.lower() for name, _ in attrs]
        if len(names) != len(set(names)):
            raise ValueError("html_duplicate_attribute")
        if any(name in {"src", "href"} for name in names):
            raise ValueError("html_external_reference_disallowed")
        self.tags[tag] = self.tags.get(tag, 0) + 1
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if not self.stack or self.stack[-1] != tag:
            raise ValueError("html_tag_balance_invalid")
        self.stack.pop()
        if tag == "html":
            self.root_closed = True

    def handle_data(self, data: str) -> None:
        if (not self.doctype or not self.stack or self.root_closed) and data.strip():
            raise ValueError("html_prose_outside_document")

    def handle_comment(self, data: str) -> None:
        raise ValueError("html_comment_wrapper_disallowed")

    def handle_pi(self, data: str) -> None:
        raise ValueError("html_processing_instruction_disallowed")


def normalize_html(raw: str) -> str:
    """Accept raw HTML or exactly one complete ```html fence; reject everything else."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("html_output_empty")
    match = _FENCE.fullmatch(raw)
    if match:
        document = match.group(1)
    else:
        if "```" in raw or raw.lstrip().startswith("```"):
            raise ValueError("html_markdown_fence_malformed")
        document = raw
    document = document.strip()
    if not document.lower().startswith("<!doctype html>"):
        raise ValueError("html_doctype_missing_or_prose_present")
    if not re.search(r"</html>[ \t\r\n]*\Z", document, re.IGNORECASE):
        raise ValueError("html_document_terminator_missing_or_prose_present")
    parser = _StrictDocumentParser()
    try:
        parser.feed(document)
        parser.close()
    except (AssertionError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    if not parser.doctype or parser.stack or not parser.root_closed or parser.tags.get("html") != 1:
        raise ValueError("html_document_structure_invalid")
    return document + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        cleaned = normalize_html(args.input.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        print("GENERATED_HTML_REJECTED: code=" + str(exc))
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(cleaned, encoding="utf-8")
    print("GENERATED_HTML_FORMAT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
