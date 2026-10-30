"""Vision-model steps: describing figures and transcribing equations to LaTeX.
Calls run concurrently with bounded workers, and each call is retried on failure. A block that still
fails keeps its raw text, so one bad call doesn't fail the document."""

import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI

from ..config import get_settings
from .types import ContentBlock

log = logging.getLogger("vision")

DESCRIBE_PROMPT = (
    "Describe this image so it can be found by search. Transcribe any visible text exactly. "
    "If it is a chart, table, or diagram, state what it shows and the key values. "
    "Be factual and concise. Do not speculate about anything not visible."
)
LATEX_PROMPT = (
    "Transcribe the mathematical expression in this image as LaTeX. "
    "Output only the LaTeX, without $ delimiters or commentary."
)


def vision_llm():
    s = get_settings()
    if not s.mistral_api_key:
        raise RuntimeError("ENABLE_VLM is true but MISTRAL_API_KEY is not set")
    return ChatMistralAI(model=s.vision_model, api_key=s.mistral_api_key, temperature=0)


def _ask(llm, prompt: str, block: ContentBlock, retries: int = 3) -> str:
    b64 = base64.b64encode(block.image_bytes).decode("ascii")
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": f"data:{block.image_mime};base64,{b64}"},
    ])
    for attempt in range(retries):
        try:
            return str(llm.invoke([message]).content).strip()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return ""


def enrich_blocks(blocks: list[ContentBlock]) -> None:
    """Fills in image descriptions and equation LaTeX, in place and concurrently."""
    s = get_settings()
    targets = [b for b in blocks if b.image_bytes is not None and (
        (b.modality == "image" and not b.text) or b.modality == "equation")]
    if not targets:
        return
    llm = vision_llm()

    def work(block: ContentBlock) -> None:
        try:
            if block.modality == "image":
                block.text = _ask(llm, DESCRIBE_PROMPT, block)
            else:
                raw = block.text
                latex = _ask(llm, LATEX_PROMPT, block)
                block.meta["raw"] = raw
                block.meta["latex"] = latex
                block.text = f"{latex}\n{raw}" if latex else raw
        except Exception as e:
            log.warning("vision call failed for %s on page %d: %s", block.modality, block.page, e)

    with ThreadPoolExecutor(max_workers=max(1, s.vlm_workers)) as pool:
        list(pool.map(work, targets))
