"""Knowledge graph store and retrieval.

Entities are nodes and relations are directed edges. Each node and edge records the chunk ids it
came from, so graph facts can be traced back to the text they were extracted from. Retrieval
follows LightRAG's modes:
- local:  find entities matching the question, then their relations and source chunks
- global: find relations matching the question, then their source chunks
- hybrid: both
"""

import json
import logging
import threading
from pathlib import Path

import networkx as nx
import numpy as np

from .extract import key_of

log = logging.getLogger("kg")
MAX_DESCRIPTIONS = 8


def _add_unique(values: list, value) -> None:
    if value not in values:
        values.append(value)


class KnowledgeGraph:
    def __init__(self, embedder, path: Path | None = None):
        self.embedder = embedder
        self.path = path
        self.g = nx.DiGraph()
        self.lock = threading.RLock()
        self._dirty = True
        self._ent_keys: list[str] = []
        self._ent_mat: np.ndarray | None = None
        self._rel_keys: list[tuple[str, str]] = []
        self._rel_mat: np.ndarray | None = None
        self.communities: list[dict] = []        # {id, nodes, summary, sources}
        self._comm_mat: np.ndarray | None = None
        self.community_cache: dict[str, str] = {}  # node-set hash -> summary, survives deletes
        if path is not None and path.exists():
            self._load()

    # ---- writes -------------------------------------------------------------

    def is_empty(self) -> bool:
        return self.g.number_of_nodes() == 0

    def stats(self) -> dict:
        return {"entities": self.g.number_of_nodes(), "relations": self.g.number_of_edges()}

    def add_extraction(self, doc_id: str, chunk_id: str, entities: list[dict], relations: list[dict]) -> None:
        with self.lock:
            for e in entities:
                k = key_of(e["name"])
                if not k:
                    continue
                self._ensure_node(k, e["name"], e["type"])
                node = self.g.nodes[k]
                if node["type"] == "other" and e["type"] != "other":
                    node["type"] = e["type"]
                if e["description"] and len(node["descriptions"]) < MAX_DESCRIPTIONS:
                    _add_unique(node["descriptions"], e["description"])
                _add_unique(node["sources"], chunk_id)
                _add_unique(node["docs"], doc_id)

            for r in relations:
                s, t = key_of(r["source"]), key_of(r["target"])
                if not s or not t or s == t:
                    continue
                self._ensure_node(s, r["source"], "other")
                self._ensure_node(t, r["target"], "other")
                if not self.g.has_edge(s, t):
                    self.g.add_edge(s, t, descriptions=[], keywords=[], sources=[], docs=[])
                edge = self.g.edges[s, t]
                if r["description"] and len(edge["descriptions"]) < MAX_DESCRIPTIONS:
                    _add_unique(edge["descriptions"], r["description"])
                for kw in r["keywords"]:
                    if len(edge["keywords"]) < MAX_DESCRIPTIONS:
                        _add_unique(edge["keywords"], kw)
                _add_unique(edge["sources"], chunk_id)
                _add_unique(edge["docs"], doc_id)
            self._dirty = True

    def _ensure_node(self, key: str, name: str, etype: str) -> None:
        if not self.g.has_node(key):
            self.g.add_node(key, name=name, type=etype, descriptions=[], sources=[], docs=[])

    def remove_document(self, doc_id: str, chunk_ids: list[str]) -> None:
        """Removes a document's chunks from every node and edge. Nodes and edges with no remaining sources are dropped."""
        gone = set(chunk_ids)
        with self.lock:
            for k in list(self.g.nodes):
                node = self.g.nodes[k]
                node["sources"] = [c for c in node["sources"] if c not in gone]
                node["docs"] = [d for d in node["docs"] if d != doc_id]
                if not node["sources"]:
                    self.g.remove_node(k)
            for s, t in list(self.g.edges):
                edge = self.g.edges[s, t]
                edge["sources"] = [c for c in edge["sources"] if c not in gone]
                edge["docs"] = [d for d in edge["docs"] if d != doc_id]
                if not edge["sources"]:
                    self.g.remove_edge(s, t)
            self._dirty = True

    def save(self) -> None:
        if self.path is None:
            return
        with self.lock:
            data = {"graph": nx.node_link_data(self.g, edges="edges"), "community_cache": self.community_cache}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.path)

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if "graph" in data:  # current format
            self.community_cache = data.get("community_cache", {})
            data = data["graph"]
        self.g = nx.node_link_graph(data, edges="edges", directed=True)
        self._dirty = True

    # ---- retrieval ----------------------------------------------------------

    def _node_text(self, k: str) -> str:
        n = self.g.nodes[k]
        return f"{n['name']} ({n['type']}). " + " ".join(n["descriptions"])

    def _edge_text(self, s: str, t: str) -> str:
        e = self.g.edges[s, t]
        return (f"{self.g.nodes[s]['name']} -> {self.g.nodes[t]['name']}: "
                f"{' '.join(e['descriptions'])} Keywords: {', '.join(e['keywords'])}")

    def _ensure_index(self) -> None:
        if not self._dirty:
            return
        self._ent_keys = list(self.g.nodes)
        self._rel_keys = list(self.g.edges)
        self._ent_mat = self._embed([self._node_text(k) for k in self._ent_keys])
        self._rel_mat = self._embed([self._edge_text(s, t) for s, t in self._rel_keys])
        self._dirty = False

    def _embed(self, texts: list[str]) -> np.ndarray | None:
        if not texts:
            return None
        return np.asarray(self.embedder.encode(texts, normalize_embeddings=True, batch_size=32), dtype=np.float32)

    def _relation_fact(self, s: str, t: str) -> str:
        e = self.g.edges[s, t]
        desc = " ".join(e["descriptions"]) or ", ".join(e["keywords"])
        return f"{self.g.nodes[s]['name']} -> {self.g.nodes[t]['name']}: {desc}"

    def context(self, question: str, mode: str, top_k: int = 8, max_chunks: int = 12) -> tuple[list[str], list[str]]:
        """Returns (fact strings, chunk ids ranked by how strongly the graph points to them)."""
        if mode not in ("local", "global", "hybrid"):
            return [], []
        with self.lock:
            if self.is_empty():
                return [], []
            self._ensure_index()
            qv = np.asarray(self.embedder.encode(question, normalize_embeddings=True), dtype=np.float32)
            facts: dict[str, None] = {}
            credit: dict[str, float] = {}

            def give(sources: list[str], score: float) -> None:
                for cid in sources:
                    credit[cid] = credit.get(cid, 0.0) + score

            if mode in ("local", "hybrid") and self._ent_mat is not None:
                sims = self._ent_mat @ qv
                for i in np.argsort(-sims)[:top_k]:
                    k, score = self._ent_keys[int(i)], float(sims[i])
                    n = self.g.nodes[k]
                    desc = f": {' '.join(n['descriptions'][:3])}" if n["descriptions"] else ""
                    facts[f"Entity: {n['name']} ({n['type']}){desc}"] = None
                    give(n["sources"], score)
                    for s, t, e in list(self.g.out_edges(k, data=True))[:4] + list(self.g.in_edges(k, data=True))[:4]:
                        facts[self._relation_fact(s, t)] = None
                        give(e["sources"], score * 0.5)

            if mode in ("global", "hybrid") and self._rel_mat is not None:
                sims = self._rel_mat @ qv
                for i in np.argsort(-sims)[:top_k]:
                    s, t = self._rel_keys[int(i)]
                    facts[self._relation_fact(s, t)] = None
                    give(self.g.edges[s, t]["sources"], float(sims[i]))

        ranked = sorted(credit, key=credit.get, reverse=True)[:max_chunks]
        return list(facts), ranked

    def refresh_communities(self, summarize=None, min_size: int = 3, max_communities: int = 30) -> None:
        """Clusters the graph with the Louvain method and summarizes each cluster. Summaries are cached by the
        cluster's node set, so an unchanged cluster is not summarized again. Pass summarize=None to reuse
        cached summaries only."""
        import hashlib

        from networkx.algorithms.community import louvain_communities

        with self.lock:
            undirected = self.g.to_undirected()
            clusters = louvain_communities(undirected, seed=7) if undirected.number_of_nodes() else []
            clusters = sorted((c for c in clusters if len(c) >= min_size), key=len, reverse=True)[:max_communities]
            communities = []
            for nodes in clusters:
                names = sorted(nodes)
                key = hashlib.sha1("|".join(names).encode()).hexdigest()
                summary = self.community_cache.get(key)
                if summary is None and summarize is not None:
                    facts = [self._node_text(k) for k in names][:40]
                    summary = (summarize(facts) or "").strip()
                    if summary:
                        self.community_cache[key] = summary
                if not summary:
                    continue
                sources: list[str] = []
                for k in names:
                    for cid in self.g.nodes[k]["sources"]:
                        if cid not in sources:
                            sources.append(cid)
                communities.append({"id": key[:12], "nodes": [self.g.nodes[k]["name"] for k in names],
                                    "summary": summary, "sources": sources})
            self.communities = communities
            self._comm_mat = self._embed([c["summary"] for c in communities]) if communities else None
        self.save()

    def community_context(self, question: str, top_k: int = 3, max_chunks: int = 12) -> tuple[list[str], list[str]]:
        """GraphRAG retrieval: the summaries of the clusters closest to the question, plus their source chunks."""
        with self.lock:
            if not self.communities or self._comm_mat is None:
                return [], []
            qv = np.asarray(self.embedder.encode(question, normalize_embeddings=True), dtype=np.float32)
            sims = self._comm_mat @ qv
            chosen = [self.communities[int(i)] for i in np.argsort(-sims)[:top_k]]
        facts = [f"Community summary: {c['summary']}" for c in chosen]
        chunk_ids: list[str] = []
        for c in chosen:
            for cid in c["sources"]:
                if cid not in chunk_ids:
                    chunk_ids.append(cid)
        return facts, chunk_ids[:max_chunks]
