"""Pipeline options: which RAG mode to run, which retrieval techniques and rerankers to apply.
Old mode names from the first version are accepted as aliases."""

from dataclasses import dataclass

MODES = ("traditional", "crag", "self_rag", "lightrag", "agentic", "graphrag")
LIGHTRAG_MODES = ("local", "global", "hybrid")
TECHNIQUES = ("hybrid", "query_rewriting", "contextual_compression", "parent_document",
              "sentence_window", "auto_merging")
RERANKERS = ("cross_encoder", "flashrank", "cohere", "bi_encoder")

# name used by earlier versions -> (mode, lightrag sub-mode)
ALIASES = {"naive": ("traditional", None), "local": ("lightrag", "local"),
           "global": ("lightrag", "global"), "hybrid": ("lightrag", "hybrid")}


@dataclass(frozen=True)
class Options:
    mode: str
    lightrag_mode: str
    techniques: frozenset
    rerankers: tuple
    vision: bool | None = None


def make_options(mode: str = "traditional", lightrag_mode: str = "hybrid", techniques=None,
                 rerankers=None, vision: bool | None = None) -> Options:
    """Validates and normalizes options. Raises ValueError with a message the API can show."""
    if mode in ALIASES:
        mode, sub = ALIASES[mode]
        lightrag_mode = sub or lightrag_mode
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {', '.join(MODES)}")
    if lightrag_mode not in LIGHTRAG_MODES:
        raise ValueError(f"Unknown LightRAG mode '{lightrag_mode}'. Choose from: {', '.join(LIGHTRAG_MODES)}")

    techs = frozenset(techniques) if techniques is not None else frozenset({"hybrid"})
    bad = sorted(techs - set(TECHNIQUES))
    if bad:
        raise ValueError(f"Unknown technique(s): {', '.join(bad)}")

    rr = tuple(dict.fromkeys(rerankers)) if rerankers else ("cross_encoder",)
    bad = sorted(set(rr) - set(RERANKERS))
    if bad:
        raise ValueError(f"Unknown reranker(s): {', '.join(bad)}")
    return Options(mode, lightrag_mode, techs, rr, vision)
