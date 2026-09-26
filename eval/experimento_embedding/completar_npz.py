#!/usr/bin/env python3
"""Realinha os .npz do BGE com os .jsonl atuais, calculando só o que falta.

O PROBLEMA QUE ISTO RESOLVE. Embutir as quatro condições no Colab leva tempo, e
nesse meio-tempo o corpus pode mudar — foi o que aconteceu: os `.npz` voltaram
com os ids e os textos de uma versão anterior à última limpeza. Reembutir tudo
de novo por causa disso seria desperdício: das 15.864 entradas necessárias,
15.828 já estavam calculadas, com o texto exatamente igual.

O que torna o reaproveitamento possível é o `embedding_text_hash` — hash do
TEXTO de embedding, sem o id. O id muda quando o texto muda (é conteúdo
endereçável), então casar por id não reaproveitaria nada; casar por texto
reaproveita tudo que de fato não mudou. Foi para isto que esse hash existe.

O QUE ELE FAZ

  1. Junta os quatro `.npz` num cache só, indexado por hash do texto. Vetor de
     qualquer condição serve para qualquer outra — T400 e PC compartilham muito
     filho idêntico.
  2. Descobre quais textos dos `.jsonl` atuais não estão no cache.
  3. Calcula só esses, com o BGE-m3 local (CPU).
  4. Regrava os quatro `.npz` na ORDEM e com os IDS dos `.jsonl` atuais.

Faz backup do que estava lá antes de sobrescrever.

SÓ SERVE PARA O BGE. O Gemini tem o seu próprio caminho com retomada
(`gemini_embutir.py`), que já lê o cache e recalcula o que falta sozinho.

Uso:
  python eval/experimento_embedding/completar_npz.py            # relatório
  python eval/experimento_embedding/completar_npz.py --aplicar  # grava
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
PROJECT_ROOT = AQUI.parents[1]
DADOS = AQUI / "dados"
RESULTADOS = AQUI / "resultados"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

from embedding_text import (EMBEDDING_TEXT_PROFILE, build_embedding_text,  # noqa: E402
                            embedding_text_hash)

CONDICOES = ("T3500", "T1200", "T400", "PC")
MODELO = "BAAI/bge-m3"
DIM = 1024


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aplicar", action="store_true", help="regrava os .npz")
    args = ap.parse_args()

    registros: dict[str, list[dict]] = {}
    hashes: dict[str, list[str]] = {}
    texto_de: dict[str, str] = {}
    for nome in CONDICOES:
        caminho = DADOS / f"{nome}.jsonl"
        if not caminho.exists():
            raise SystemExit(f"falta {caminho}")
        regs = [json.loads(l) for l in caminho.open(encoding="utf-8") if l.strip()]
        registros[nome] = regs
        hs = []
        for r in regs:
            h = embedding_text_hash(r)
            hs.append(h)
            texto_de.setdefault(h, build_embedding_text(r))
        hashes[nome] = hs

    # ── cache: todo vetor de todo .npz, por hash do texto ───────────────────
    cache: dict[str, np.ndarray] = {}
    print("cache montado a partir de:")
    for nome in CONDICOES:
        arq = RESULTADOS / f"emb_{nome}.npz"
        if not arq.exists():
            print(f"   {arq.name}  AUSENTE")
            continue
        d = np.load(arq, allow_pickle=True)
        if str(d["embedding_model"]) != MODELO:
            raise SystemExit(f"{arq.name} não é do {MODELO}: {d['embedding_model']}")
        if str(d["embedding_text_profile"]) != EMBEDDING_TEXT_PROFILE:
            raise SystemExit(
                f"{arq.name} foi gerado com outro perfil de texto de embedding "
                f"({d['embedding_text_profile']} ≠ {EMBEDDING_TEXT_PROFILE}). "
                "Misturar perfis produz vetor incomparável — reembuta.")
        matriz = d["matrix"]
        novos = 0
        for j, h in enumerate(map(str, d["text_hash"])):
            if h not in cache:
                cache[h] = matriz[j]
                novos += 1
        print(f"   {arq.name:22s} {matriz.shape}  +{novos} hashes")

    necessarios = list(texto_de)
    faltam = [h for h in necessarios if h not in cache]
    print(f"\n  textos distintos necessários : {len(necessarios)}")
    print(f"  já no cache                  : {len(necessarios) - len(faltam)}"
          f"  ({1 - len(faltam)/len(necessarios):.1%})")
    print(f"  a calcular                   : {len(faltam)}")
    for nome in CONDICOES:
        n = sum(1 for h in hashes[nome] if h not in cache)
        print(f"     {nome:6s} {len(registros[nome]):6d} trechos, {n} sem vetor")

    if not args.aplicar:
        print("\n(relatório apenas — use --aplicar para calcular e regravar)")
        return

    # ── calcular só o que falta ─────────────────────────────────────────────
    if faltam:
        print(f"\n  carregando {MODELO} (CPU)...", flush=True)
        from sentence_transformers import SentenceTransformer
        modelo = SentenceTransformer(MODELO, device="cpu")
        t0 = time.perf_counter()
        vetores = modelo.encode([texto_de[h] for h in faltam],
                                normalize_embeddings=True, batch_size=8,
                                show_progress_bar=True).astype("float32")
        for h, v in zip(faltam, vetores):
            cache[h] = v
        print(f"  {len(faltam)} vetores em {(time.perf_counter()-t0)/60:.1f} min")

    # ── regravar na ordem e com os ids atuais ───────────────────────────────
    carimbo = datetime.now().strftime("%Y%m%d-%H%M%S")
    print()
    for nome in CONDICOES:
        hs = hashes[nome]
        matriz = np.stack([cache[h] for h in hs]).astype("float32")
        normas = np.linalg.norm(matriz, axis=1)
        assert matriz.shape == (len(hs), DIM), f"forma inesperada: {matriz.shape}"
        assert np.allclose(normas, 1.0, atol=1e-3), "vetor não normalizado"
        assert not np.isnan(matriz).any(), "vetor com NaN"

        destino = RESULTADOS / f"emb_{nome}.npz"
        if destino.exists():
            shutil.copy2(destino, f"{destino}.bak-{carimbo}")
        np.savez_compressed(
            destino,
            matrix=matriz,
            ids=np.array([str(r["id"]) for r in registros[nome]], dtype=object),
            text_hash=np.array(hs, dtype=object),
            embedding_text_profile=np.array(EMBEDDING_TEXT_PROFILE),
            embedding_model=np.array(MODELO),
        )
        print(f"  ✓ {destino.name}  {matriz.shape}  "
              f"norma {normas.min():.4f}–{normas.max():.4f}")
    print(f"\n  backups com sufixo .bak-{carimbo}")


if __name__ == "__main__":
    main()
