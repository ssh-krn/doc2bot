def table_to_markdown(rows: list[list[object]]) -> str:
    """Render rows as a Markdown table. The first non-empty row is treated as the header."""

    def cell(value: object) -> str:
        if value is None:
            return ""
        return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()

    cleaned = [[cell(c) for c in row] for row in rows if row and any(cell(c) for c in row)]
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    lines = ["| " + " | ".join(cleaned[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in cleaned[1:]]
    return "\n".join(lines)
