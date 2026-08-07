#!/usr/bin/env python3
"""Validate the GitHub Pages source without third-party dependencies."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LINK = re.compile(r"(?:\[[^]]*\]\(([^) ]+)(?:\s+[^)]*)?\)|(?:href|src)=[\"']([^\"']+))")
LIQUID_BASE = "{{ site.baseurl }}"
CONFIG = (DOCS / "_config.yml").read_text(encoding="utf-8")
BASEURL_MATCH = re.search(r"^baseurl:\s*[\"']?([^\"'\n]+)", CONFIG, re.MULTILINE)
BASEURL = BASEURL_MATCH.group(1).strip() if BASEURL_MATCH else ""


def page_target(path: Path) -> Path:
    if path.suffix:
        return path
    return path / "index.md"


def resolve(source: Path, raw: str) -> Path | None:
    raw = raw.strip("<>\"'")
    if not raw or raw.startswith(("#", "mailto:", "tel:", "data:", "javascript:")):
        return None
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc:
        return None
    target = unquote(parsed.path)
    if "{{" in target and not target.startswith(LIQUID_BASE):
        return None
    if target.startswith(LIQUID_BASE):
        target = target[len(LIQUID_BASE):]
    if BASEURL and (target == BASEURL or target.startswith(f"{BASEURL}/")):
        target = target[len(BASEURL):] or "/"
    if target.startswith("/"):
        candidate = DOCS / target.lstrip("/")
    else:
        candidate = source.parent / target
    if candidate.suffix == ".html":
        markdown = candidate.with_suffix(".md")
        if markdown.exists():
            return markdown
    if candidate.is_dir() or not candidate.suffix:
        direct = candidate.with_suffix(".md")
        if direct.exists():
            return direct
        return page_target(candidate)
    return candidate


def main() -> int:
    errors: list[str] = []
    pages = sorted(path for path in DOCS.glob("*.md") if not path.name.startswith("._"))
    if not pages:
        errors.append("docs contains no Markdown pages")
    for page in pages:
        text = page.read_text(encoding="utf-8")
        if not text.startswith("---\n") or "layout: default" not in text.split("---", 2)[1]:
            errors.append(f"{page.relative_to(ROOT)}: missing default-layout front matter")
        if not re.search(r"^title:\s+.+$", text, re.MULTILINE):
            errors.append(f"{page.relative_to(ROOT)}: missing title")
        for match in LINK.finditer(text):
            raw = match.group(1) or match.group(2)
            target = resolve(page, raw)
            if target is not None and not target.exists():
                errors.append(f"{page.relative_to(ROOT)}: broken link {raw!r}")

    required = [
        DOCS / "_config.yml",
        DOCS / "_layouts/default.html",
        DOCS / "assets/css/style.css",
        DOCS / "assets/js/site.js",
        DOCS / "assets/img/mark.svg",
    ]
    errors.extend(f"missing required file: {path.relative_to(ROOT)}" for path in required if not path.exists())

    if errors:
        print("Documentation validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Documentation validation passed: {len(pages)} pages checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
