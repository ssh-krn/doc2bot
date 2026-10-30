"""Entity and relation extraction from a passage. The model is asked for JSON; the parser tolerates
code fences, leading prose, and malformed items, and drops anything it can't use."""

import json
import re

ENTITY_TYPES = ("person", "organization", "product", "location", "event", "concept", "date", "other")

PROMPT = """You extract a knowledge graph from a passage.
Return ONLY a JSON object, with no prose, in this shape:
{"entities": [{"name": "...", "type": "...", "description": "..."}],
 "relations": [{"source": "...", "target": "...", "description": "...", "keywords": ["..."]}]}

Rules:
- Entity names are the proper names or defined terms as written in the passage.
- type is one of: person, organization, product, location, event, concept, date, other.
- Include only facts the passage states. Do not add outside knowledge.
- Every relation's source and target must also appear in entities.
- If the passage has no entities, return {"entities": [], "relations": []}.

Passage:
<<<
PASSAGE_TEXT
>>>"""


def build_prompt(passage: str) -> str:
    # Replace a placeholder instead of str.format, so braces inside the passage are safe.
    return PROMPT.replace("PASSAGE_TEXT", passage)


def key_of(name: str) -> str:
    """Canonical key for matching entity names: case, punctuation and spacing are ignored."""
    return " ".join(re.sub(r"[^\w\s]", " ", name.lower()).split())


def _clean(value: object) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def parse_extraction(raw: str) -> tuple[list[dict], list[dict]]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return [], []
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return [], []
    if not isinstance(data, dict):
        return [], []

    entities: list[dict] = []
    seen: set[str] = set()
    for item in data.get("entities") or []:
        if not isinstance(item, dict):
            continue
        name = _clean(item.get("name"))
        if not key_of(name) or key_of(name) in seen:
            continue
        etype = _clean(item.get("type")).lower()
        entities.append({
            "name": name,
            "type": etype if etype in ENTITY_TYPES else "other",
            "description": _clean(item.get("description")),
        })
        seen.add(key_of(name))

    relations: list[dict] = []
    for item in data.get("relations") or []:
        if not isinstance(item, dict):
            continue
        source, target = _clean(item.get("source")), _clean(item.get("target"))
        if not key_of(source) or not key_of(target):
            continue
        keywords = item.get("keywords") if isinstance(item.get("keywords"), list) else []
        relations.append({
            "source": source,
            "target": target,
            "description": _clean(item.get("description")),
            "keywords": [_clean(k) for k in keywords if _clean(k)][:8],
        })
        for name in (source, target):  # relation endpoints must exist as entities
            if key_of(name) not in seen:
                entities.append({"name": name, "type": "other", "description": ""})
                seen.add(key_of(name))
    return entities, relations
