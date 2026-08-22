"""
eval/lib/embedders.py
═════════════════════
Interface única de embedding para o Cenário B (dense) e para a fusão.

Implementações:
  • GeminiEmbedder              — gemini-embedding-001 (3072d), caminho de produção
  • SentenceTransformerEmbedder — BGE-m3 (multilíngue) / e5 / etc.

Uso:
    emb = build_embedder("bge_m3", cfg)
    doc_vecs = emb.embed_documents([...])   # (n, dim) np.float32, L2-normalizado
    q_vec    = emb.embed_query("...")       # (dim,)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from .common import get_env


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        n = np.linalg.norm(mat)
        return mat / n if n > 0 else mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class BaseEmbedder:
    name: str = "base"
    dim: int = 0

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


# ── Gemini ────────────────────────────────────────────────────────────────--

class GeminiEmbedder(BaseEmbedder):
    """gemini-embedding-001 via google-genai. Replica o caminho de produção
    (task_type RETRIEVAL_DOCUMENT/QUERY, normalização, respeito a rate limit)."""

    def __init__(self, ecfg: dict):
        from google import genai
        self.name = "gemini"
        self.model = ecfg.get("model", "gemini-embedding-001")
        self.dim = int(ecfg.get("dim", 3072))
        self.task_doc = ecfg.get("task_type_doc", "RETRIEVAL_DOCUMENT")
        self.task_query = ecfg.get("task_type_query", "RETRIEVAL_QUERY")
        self.normalize = ecfg.get("normalize", True)
        self.batch_size = int(ecfg.get("batch_size", 20))
        self.sleep = float(ecfg.get("sleep_between_batches", 8))
        self._client = genai.Client(api_key=get_env("GOOGLE_GENAI_API_KEY", required=True))

    def _embed(self, texts: List[str], task_type: str) -> np.ndarray:
        from google.genai import types
        out: List[List[float]] = []
        for i in tqdm(range(0, len(texts), self.batch_size),
                      desc=f"[gemini:{task_type}]", leave=False):
            batch = texts[i:i + self.batch_size]
            resp = self._client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            out.extend([e.values for e in resp.embeddings])
            if self.sleep and i + self.batch_size < len(texts):
                time.sleep(self.sleep)
        arr = np.array(out, dtype=np.float32)
        return _l2_normalize(arr) if self.normalize else arr

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._embed(texts, self.task_doc)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], self.task_query)[0]


# ── Sentence-Transformers (BGE-m3, e5, ...) ─────────────────────────────────--

class SentenceTransformerEmbedder(BaseEmbedder):
    """Modelos locais via sentence-transformers.

    Alguns modelos (ex.: e5) exigem instrução no lado da query; o BGE-m3 não.
    Isso é controlado por `query_instruction` na config (vazio = sem instrução)."""

    def __init__(self, name: str, ecfg: dict):
        from sentence_transformers import SentenceTransformer
        import torch

        self.name = name
        self.model_id = ecfg["model"]
        self.normalize = ecfg.get("normalize", True)
        self.query_instruction = ecfg.get("query_instruction", "") or ""
        self.batch_size = int(ecfg.get("batch_size", 8))

        device = ecfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[{name}] Carregando '{self.model_id}' em {device} "
              f"(pode baixar ~vários GB na 1ª vez)...")
        self._model = SentenceTransformer(self.model_id, device=device,
                                          trust_remote_code=True)
        self.dim = self._model.get_sentence_embedding_dimension()
        print(f"[{name}] dim = {self.dim}")

    def _encode(self, texts: List[str]) -> np.ndarray:
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 32,
        )
        return emb.astype(np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_instruction + text])[0]


# ── Precomputado (offline / Colab) ───────────────────────────────────────────

class PrecomputedEmbedder(BaseEmbedder):
    """Usa vetores pré-computados por eval/tools/embed_offline.py — NÃO carrega modelo.
    A matriz de DOCUMENTOS vem do cache do DenseRetriever (dense_<name>_<N>.npz);
    aqui resolvemos apenas as QUERIES (dense_<name>_queries.npz)."""

    def __init__(self, name: str, indexes_dir: Path):
        self.name = name
        qp = Path(indexes_dir) / f"dense_{name}_queries.npz"
        if not qp.exists():
            raise FileNotFoundError(
                f"Vetores de query ausentes: {qp}\n  → rode eval/tools/embed_offline.py "
                f"--name {name} (Colab/GPU) e copie o .npz para results/indexes/.")
        data = np.load(qp, allow_pickle=True)
        mat = data["matrix"].astype(np.float32)
        self.dim = int(mat.shape[1])
        self._qmap = {str(t): mat[i] for i, t in enumerate(data["qtexts"])}

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError(
            "Modo offline: a matriz de documentos vem do cache dense_<name>_<N>.npz. "
            "Garanta que o .npz de documentos está em results/indexes/ (DenseRetriever o lê).")

    def embed_query(self, text: str) -> np.ndarray:
        v = self._qmap.get(str(text))
        if v is None:
            raise KeyError(f"Query sem vetor pré-computado: {text[:70]!r}. "
                           f"Re-rode embed_offline.py com o gold set atual.")
        return v.astype(np.float32)


# ── Fábrica ──────────────────────────────────────────────────────────────────

def build_embedder(name: str, cfg: dict) -> BaseEmbedder:
    ecfg = cfg["embedders"][name]
    if ecfg.get("offline"):
        from .common import resolve
        idx = resolve(cfg["paths"]["results_dir"]) / "indexes"
        return PrecomputedEmbedder(name, idx)
    etype = ecfg.get("type")
    if etype == "gemini":
        return GeminiEmbedder(ecfg)
    if etype == "sentence_transformers":
        return SentenceTransformerEmbedder(name, ecfg)
    raise ValueError(f"Tipo de embedder desconhecido para '{name}': {etype}")
