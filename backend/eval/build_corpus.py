"""Renders the evaluation corpus. Source text lives in eval/corpus_src/*.txt; pages are separated by a line
containing %%PAGE%%. Each source file becomes one PDF in eval/corpus/, with one text page per source page.

Run from backend/: python -m eval.build_corpus
"""

from pathlib import Path

import pymupdf

SRC = Path("eval/corpus_src")
OUT = Path("eval/corpus")
MARK = "%%PAGE%%"


def render(text: str, path: Path) -> None:
    doc = pymupdf.open()
    for page_text in (p.strip() for p in text.split(MARK)):
        page = doc.new_page(width=595, height=842)  # A4
        rect = pymupdf.Rect(60, 60, 535, 782)
        left = page.insert_textbox(rect, page_text, fontsize=10, fontname="helv", align=0)
        if left < 0:
            raise SystemExit(f"{path.name}: page text overflows the page; shorten it")
    doc.save(path)
    doc.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.txt"))
    if not files:
        raise SystemExit(f"No source files in {SRC}")
    for src in files:
        out = OUT / (src.stem + ".pdf")
        render(src.read_text(encoding="utf-8"), out)
        print(f"built {out} ({len(src.read_text(encoding='utf-8').split(MARK))} pages)")


if __name__ == "__main__":
    main()
