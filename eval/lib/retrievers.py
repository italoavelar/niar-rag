"""
eval/lib/retrievers.py
══════════════════════
Cenários de recuperação sobre o corpus de chunks (jurídico-saúde, PT-BR):

  A) BM25Retriever         — léxico, com pré-processamento PT (stopwords + RSLP)
  B) DenseRetriever        — semântico, embedder pluggável (gemini | bge-m3 | ...)
     QdrantDenseRetriever  — variante que usa a coleção Qdrant de produção (gemini)
  C) rrf()                 — fusão Reciprocal Rank Fusion de A + B

Todos expõem run_queries(queries) -> {qid: [chunk_id ordenado]}.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .common import ensure_dir, get_env
from .embedders import BaseEmbedder


# ════════════════════════════════════════════════════════════════════════════
# Pré-processamento PT-BR
# ════════════════════════════════════════════════════════════════════════════

_TOKEN_RE = re.compile(r"[a-zA-Zà-úÀ-Ú0-9]+")


def _ensure_nltk():
    import nltk
    for res, path in [("stopwords", "corpora/stopwords"), ("rslp", "stemmers/rslp")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)


# marcadores simples p/ detectar idioma do chunk (corpus é PT + EN)
_PT_MARKERS = (" de ", " que ", " não ", " são ", " para ", " com ", " dos ", " uma ")
_EN_MARKERS = (" the ", " of ", " and ", " shall ", " this ", " is ", " which ", " to ")


def detect_lang(text: str) -> str:
    """'pt' ou 'en' por contagem de marcadores. Empate → 'pt' (língua da persona)."""
    t = " " + text.lower() + " "
    pt = sum(t.count(w) for w in _PT_MARKERS)
    en = sum(t.count(w) for w in _EN_MARKERS)
    return "en" if en > pt else "pt"


class BilingualPreprocessor:
    """tokeniza -> minúsculas -> remove stopwords -> stemmer, POR IDIOMA.

    O corpus é bilíngue (PT + EN): aplicar RSLP/stopwords PT em texto inglês
    degrada o BM25 injustamente. Aqui cada texto é pré-processado com o pipeline
    da sua própria língua."""

    def __init__(self, stemmer: str = "rslp", strip_accents: bool = False):
        _ensure_nltk()
        from nltk.corpus import stopwords
        from nltk.stem import RSLPStemmer, SnowballStemmer
        self.stops = {"pt": set(stopwords.words("portuguese")),
                      "en": set(stopwords.words("english"))}
        self.stemmers = {"pt": RSLPStemmer() if stemmer == "rslp" else None,
                         "en": SnowballStemmer("english")}
        self.strip_accents = strip_accents

    @staticmethod
    def _deaccent(t: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFKD", t)
                       if not unicodedata.combining(c))

    def __call__(self, text: str, lang: str = None) -> List[str]:
        lang = lang or detect_lang(text)
        stops = self.stops.get(lang, self.stops["pt"])
        stemmer = self.stemmers.get(lang)
        out = []
        for t in _TOKEN_RE.findall(text.lower()):
            if t in stops or len(t) < 2:
                continue
            if self.strip_accents:
                t = self._deaccent(t)
            if stemmer:
                try:
                    t = stemmer.stem(t)
                except Exception:
                    pass
            out.append(t)
        return out


# ════════════════════════════════════════════════════════════════════════════
# Cenário A — BM25
# ════════════════════════════════════════════════════════════════════════════

class BM25Retriever:
    def __init__(self, corpus: Dict[str, dict], rcfg: dict,
                 cache_dir: Optional[Path] = None, rebuild: bool = False):
        from rank_bm25 import BM25Okapi
        self.pre = BilingualPreprocessor(stemmer=rcfg.get("stemmer", "rslp"))
        self.k1 = rcfg.get("k1", 1.5)
        self.b = rcfg.get("b", 0.75)
        self.use_prf = rcfg.get("use_prf", False)
        self.prf_docs = rcfg.get("prf_docs", 5)
        self.prf_terms = rcfg.get("prf_terms", 10)

        self.ids: List[str] = list(corpus.keys())
        # Cache do índice TOKENIZADO (a parte cara: stemming de milhares de chunks).
        # Chave = stemmer + nº de chunks; valida os ids p/ invalidar se o corpus mudar.
        cache = None
        if cache_dir:
            ensure_dir(cache_dir)
            cache = Path(cache_dir) / f"bm25_tokens_{rcfg.get('stemmer','rslp')}_{len(self.ids)}.pkl"
        self.tokenized = None
        if cache and cache.exists() and not rebuild:
            import pickle
            try:
                data = pickle.loads(cache.read_bytes())
                if data.get("ids") == self.ids:
                    self.tokenized = data["tokenized"]
                    print(f"[BM25] índice em cache ({len(self.ids)} chunks) ← {cache.name}")
            except Exception:
                self.tokenized = None
        if self.tokenized is None:
            print(f"[BM25] Tokenizando {len(self.ids)} chunks (PT/EN por idioma)...")
            self.tokenized, langs = [], []
            for cid in tqdm(self.ids, leave=False):
                txt = corpus[cid]["text"]
                lg = corpus[cid].get("metadata", {}).get("lang") or detect_lang(txt)
                langs.append(lg)
                self.tokenized.append(self.pre(txt, lg))
            from collections import Counter
            print(f"[BM25] idiomas no índice: {dict(Counter(langs))}")
            if cache:
                import pickle
                cache.write_bytes(pickle.dumps({"ids": self.ids, "tokenized": self.tokenized}))
                print(f"[BM25] índice salvo → {cache.name}")
        self.bm25 = BM25Okapi(self.tokenized, k1=self.k1, b=self.b)

    def _tokenize_query(self, query: str, translations: dict = None) -> List[str]:
        """A consulta é PT (persona), mas o índice é bilíngue.
        - termos PT (do RSLP/stopwords PT) casam os chunks em português;
        - se houver TRADUÇÃO da query (CLIR), tokenizamos o texto EN traduzido no
          pipeline inglês → casa os chunks em inglês ("prontuário" → "record");
        - sem tradução, caímos no pipeline EN sobre a própria query PT, que só
          resgata siglas/cognatos (LGPD, GDPR, software)."""
        toks = self.pre(query, "pt")
        en_src = (translations or {}).get(query)
        toks += self.pre(en_src, "en") if en_src else self.pre(query, "en")
        return list(dict.fromkeys(toks))       # dedup preservando ordem

    def _expand_prf(self, q_tokens: List[str]) -> List[str]:
        scores = self.bm25.get_scores(q_tokens)
        top = np.argsort(scores)[::-1][:self.prf_docs]
        freq: Dict[str, int] = {}
        for idx in top:
            for t in self.tokenized[idx]:
                freq[t] = freq.get(t, 0) + 1
        extra = sorted(freq, key=freq.get, reverse=True)[:self.prf_terms]
        return q_tokens + extra

    def search(self, query: str, top_k: int = 100,
               translations: dict = None) -> List[Tuple[str, float]]:
        q = self._tokenize_query(query, translations)
        if self.use_prf:
            q = self._expand_prf(q)
        scores = self.bm25.get_scores(q)
        top = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], float(scores[i])) for i in top]

    def run_queries(self, queries: Sequence[Tuple[str, str]], top_k: int = 100,
                    translations: dict = None) -> Dict[str, List[str]]:
        tag = " (query traduzida)" if translations else ""
        out = {}
        for qid, qtext in tqdm(queries, desc=f"[BM25] buscando{tag}", leave=False):
            out[qid] = [cid for cid, _ in self.search(qtext, top_k, translations)]
        return out


# ════════════════════════════════════════════════════════════════════════════
# Cenário B — Dense (in-memory, embedder pluggável, com cache em disco)
# ════════════════════════════════════════════════════════════════════════════

class DenseRetriever:
    def __init__(self, corpus: Dict[str, dict], embedder: BaseEmbedder,
                 cache_dir: Optional[Path] = None, rebuild: bool = False):
        self.embedder = embedder
        self.ids: List[str] = list(corpus.keys())
        self._matrix: Optional[np.ndarray] = None

        cache = None
        if cache_dir:
            ensure_dir(cache_dir)
            # nome inclui o nº de docs p/ não confundir corpus completo vs subconjunto (rerank)
            cache = Path(cache_dir) / f"dense_{embedder.name}_{len(self.ids)}.npz"

        if cache and cache.exists() and not rebuild:
            data = np.load(cache, allow_pickle=True)
            self.ids = [str(x) for x in data["ids"]]   # evita np.str_ vazar adiante
            self._matrix = data["matrix"].astype(np.float32)
            print(f"[Dense:{embedder.name}] cache carregado ({self._matrix.shape}) ← {cache.name}")
        else:
            texts = [corpus[cid]["text"] for cid in self.ids]
            print(f"[Dense:{embedder.name}] Embutindo {len(texts)} chunks...")
            self._matrix = embedder.embed_documents(texts).astype(np.float32)
            if cache:
                np.savez_compressed(cache, matrix=self._matrix, ids=np.array(self.ids))
                print(f"[Dense:{embedder.name}] cache salvo → {cache.name}")

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        q = self.embedder.embed_query(query).astype(np.float32)
        scores = self._matrix @ q                 # vetores L2-normalizados → cosseno
        top = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], float(scores[i])) for i in top]

    def rerank(self, query: str, candidate_ids: Sequence[str], top_k: int = 10
               ) -> List[Tuple[str, float]]:
        """Re-ranqueia APENAS os candidate_ids (cascata BM25→dense). Os ids precisam
        estar no índice (em modo rerank, embuta o corpus restrito aos candidatos)."""
        idmap = {cid: i for i, cid in enumerate(self.ids)}
        rows = [(cid, idmap[cid]) for cid in candidate_ids if cid in idmap]
        if not rows:
            return []
        q = self.embedder.embed_query(query).astype(np.float32)
        sub = self._matrix[[i for _, i in rows]]
        scores = sub @ q
        order = np.argsort(scores)[::-1][:top_k]
        return [(rows[o][0], float(scores[o])) for o in order]

    def run_queries(self, queries: Sequence[Tuple[str, str]], top_k: int = 100
                    ) -> Dict[str, List[str]]:
        out = {}
        for qid, qtext in tqdm(queries, desc=f"[Dense:{self.embedder.name}] buscando",
                               leave=False):
            out[qid] = [cid for cid, _ in self.search(qtext, top_k)]
        return out


class QdrantDenseRetriever:
    """Variante de produção: consulta a coleção Qdrant existente (embeddings Gemini
    já indexados por build_vectorstore.py). Mapeia o resultado para chunk_id via
    payload['id_original']."""

    def __init__(self, embedder: BaseEmbedder, collection: str):
        from urllib.parse import urlsplit
        self.embedder = embedder
        self.collection = collection
        sp = urlsplit(get_env("QDRANT_URL", required=True).rstrip("/"))
        scheme = sp.scheme or "https"
        port = sp.port or 6333          # Qdrant Cloud serve a REST na 6333 (não na 443)
        self.url = f"{scheme}://{sp.hostname}:{port}"
        self.api_key = get_env("QDRANT_API_KEY", required=True)

    def _headers(self):
        return {"api-key": self.api_key, "Content-Type": "application/json"}

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        import requests
        q = self.embedder.embed_query(query).tolist()
        base = f"{self.url}/collections/{self.collection}/points"
        # tenta o endpoint clássico (/search) e, se ausente, o novo (/query)
        attempts = (
            (f"{base}/search", {"vector": q, "limit": top_k, "with_payload": True}),
            (f"{base}/query",  {"query": q,  "limit": top_k, "with_payload": True}),
        )
        last = None
        for endpoint, body in attempts:
            resp = requests.post(endpoint, headers=self._headers(), json=body, timeout=60)
            if resp.status_code == 404:
                last = resp; continue
            if resp.status_code >= 400:
                raise RuntimeError(f"Qdrant {resp.status_code} em {endpoint}: {resp.text[:200]}")
            result = resp.json().get("result")
            pts = result.get("points", []) if isinstance(result, dict) else (result or [])
            out = []
            for p in pts:
                cid = (p.get("payload") or {}).get("id_original")
                if cid is not None:
                    out.append((str(cid), float(p.get("score", 0.0))))
            return out
        raise RuntimeError(
            "Qdrant: nenhum endpoint de busca respondeu (404 em /search e /query). "
            f"Confira a coleção '{self.collection}' e o QDRANT_URL. URL tentada: "
            f"{last.url if last is not None else self.url}")

    def run_queries(self, queries: Sequence[Tuple[str, str]], top_k: int = 100
                    ) -> Dict[str, List[str]]:
        out = {}
        for qid, qtext in tqdm(queries, desc="[Qdrant] buscando", leave=False):
            out[qid] = [cid for cid, _ in self.search(qtext, top_k)]
        return out


# ════════════════════════════════════════════════════════════════════════════
# Cenário C — Fusão RRF
# ════════════════════════════════════════════════════════════════════════════

def rrf(rankings: Dict[str, Dict[str, List[str]]], k: int = 30) -> Dict[str, List[str]]:
    """Reciprocal Rank Fusion. rankings = {sistema: {qid: [chunk_id]}}."""
    all_qids: set = set()
    for sysr in rankings.values():
        all_qids.update(sysr.keys())
    fused = {}
    for qid in all_qids:
        scores: Dict[str, float] = {}
        for sysr in rankings.values():
            for rank, cid in enumerate(sysr.get(qid, [])):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        fused[qid] = sorted(scores, key=scores.get, reverse=True)
    return fused
