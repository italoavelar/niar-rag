"""
eval/lib/common.py
══════════════════
Utilidades compartilhadas: I/O UTF-8 (Windows), config, corpus, seed, env, retry.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

# Diretórios de referência (eval/ e a raiz do projeto RAG)
EVAL_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EVAL_DIR.parent


def setup_io() -> None:
    """Força UTF-8 no stdout/stderr — o console do Windows usa cp1252 e quebra
    com acentos/emojis. Chame no início de todo script driver."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")        # Python 3.7+
        except Exception:
            pass


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


# ── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str | Path | None = None) -> dict:
    import yaml
    cfg_path = Path(path) if path else (EVAL_DIR / "config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_eval_dir"] = str(EVAL_DIR)
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve um caminho do config relativo ao diretório eval/."""
    p = Path(path)
    return p if p.is_absolute() else (EVAL_DIR / p).resolve()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Env / .env ─────────────────────────────────────────────────────────────--

_dotenv_loaded = False


def get_env(key: str, required: bool = False) -> str | None:
    """Lê variável de ambiente, carregando o .env da raiz do RAG na 1ª chamada.

    Trata o apelido GOOGLE_GENAI_API_KEY <-> GOOGLE_API_KEY (o código de produção
    usa GOOGLE_GENAI_API_KEY, mas o .env define GOOGLE_API_KEY)."""
    global _dotenv_loaded
    if not _dotenv_loaded:
        try:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")
        except Exception:
            pass
        _dotenv_loaded = True

    val = os.getenv(key)
    if val is None and key == "GOOGLE_GENAI_API_KEY":
        val = os.getenv("GOOGLE_API_KEY")
    if val is None and key == "GOOGLE_API_KEY":
        val = os.getenv("GOOGLE_GENAI_API_KEY")
    if required and not val:
        raise RuntimeError(f"Variável de ambiente obrigatória ausente: {key}")
    return val


# ── JSONL ──────────────────────────────────────────────────────────────────--

def load_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Corpus ─────────────────────────────────────────────────────────────────--

def load_corpus(path: str | Path) -> Dict[str, dict]:
    """Carrega documents.jsonl -> {chunk_id: {id, text, metadata}}."""
    corpus: Dict[str, dict] = {}
    for doc in load_jsonl(path):
        corpus[str(doc["id"])] = doc
    return corpus


def neighbors_same_doc(chunk_id: str, corpus: Dict[str, dict], window: int = 1) -> List[str]:
    """IDs de chunks vizinhos no mesmo documento (mesmo source, página±window).
    IDs têm formato '{stem}_p{page}_c{chunk}'."""
    doc = corpus.get(chunk_id)
    if not doc:
        return []
    src = doc["metadata"].get("source")
    page = doc["metadata"].get("page")
    out = []
    for cid, d in corpus.items():
        if cid == chunk_id:
            continue
        if d["metadata"].get("source") == src:
            try:
                if abs(int(d["metadata"].get("page", -999)) - int(page)) <= window:
                    out.append(cid)
            except (TypeError, ValueError):
                continue
    return out


# ── Retry para chamadas de API ───────────────────────────────────────────────

def with_retry(fn: Callable[[], Any], tries: int = 4, base_delay: float = 2.0,
               label: str = "") -> Any:
    """Executa fn() com backoff exponencial. Útil p/ rate limits de LLM/embeddings."""
    last = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:          # noqa: BLE001 — queremos capturar tudo p/ retry
            last = e
            wait = base_delay * (2 ** (attempt - 1))
            print(f"  [retry {attempt}/{tries}] {label} falhou: {e}. "
                  f"Aguardando {wait:.0f}s...")
            time.sleep(wait)
    raise RuntimeError(f"Esgotadas {tries} tentativas em {label}: {last}")
