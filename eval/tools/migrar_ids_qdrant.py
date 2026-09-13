#!/usr/bin/env python3
"""Atualiza o `id_original` no payload do Qdrant após a migração de ids.

Contexto. `eval/tools/migrar_ids_por_conteudo.py` trocou os ids de trecho de
posicionais para derivados do conteúdo no corpus, no gabarito e nos caches de
embedding. Falta a produção: a coleção do Qdrant guarda o id do trecho no
payload, no campo `id_original` (`src/build_vectorstore.py:181`), que é o que a
recuperação lê de volta (`eval/lib/retrievers.py:339`). Enquanto ele não for
atualizado, produção devolve ids antigos e nada casa com o gabarito novo.

Boa notícia: **nada precisa ser reembutido**. Os vetores continuam válidos — o
texto não mudou — e o id do ponto no Qdrant é um inteiro sequencial
(`id=index`), não o id do trecho. Então isto é um `set_payload` sobre os pontos
existentes, não uma reindexação.

Melhor ainda: o payload já carrega `texto` e `document_id`, então o id novo é
calculado a partir do próprio payload, com a MESMA função que gerou os ids do
corpus (`src/chunk_id.py`). Nada de mapa externo para desencontrar.

Colisão: dois trechos idênticos no mesmo documento receberiam o mesmo hash. O
corpus resolve com sufixo `-2`. Aqui o script detecta, avisa e resolve casando
com os ids que existem no corpus migrado — se não conseguir, para e mostra os
casos, em vez de gravar um id ambíguo.

Uso:
  python eval/tools/migrar_ids_qdrant.py                     # simulação
  python eval/tools/migrar_ids_qdrant.py --aplicar           # grava
  python eval/tools/migrar_ids_qdrant.py --colecao OUTRA     # outra coleção
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

from chunk_id import chunk_id_por_conteudo  # noqa: E402

CORPUS = PROJECT_ROOT / "data/processed/documents.jsonl"
LOTE = 256


def carregar_ids_validos() -> set[str]:
    with CORPUS.open(encoding="utf-8") as fh:
        return {json.loads(l)["id"] for l in fh if l.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Atualiza id_original no payload do Qdrant.")
    ap.add_argument("--aplicar", action="store_true", help="grava (sem isto, só simula)")
    ap.add_argument("--colecao", default=os.getenv("QDRANT_GEMINI_COLLECTION", "LEME_gemini"))
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass
    from qdrant_client import QdrantClient

    url, chave = os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")
    if not url:
        raise SystemExit("QDRANT_URL ausente no .env")

    cliente = QdrantClient(url=url, api_key=chave, timeout=120)
    validos = carregar_ids_validos()
    print("── Atualização do id_original no Qdrant ──")
    print(f"coleção: {args.colecao}   modo: {'APLICAR' if args.aplicar else 'SIMULAÇÃO'}")
    print(f"ids válidos no corpus migrado: {len(validos)}\n")

    prox, total = None, 0
    ja_ok = mudam = sem_texto = fora = 0
    exemplos, problemas = [], []
    pendentes: list[tuple] = []
    usos: Counter = Counter()

    while True:
        pontos, prox = cliente.scroll(
            collection_name=args.colecao, limit=LOTE, offset=prox,
            with_payload=True, with_vectors=False,
        )
        if not pontos:
            break
        for p in pontos:
            total += 1
            carga = p.payload or {}
            antigo = carga.get("id_original") or ""
            texto = carga.get("texto") or ""
            doc = carga.get("document_id") or ""
            if not texto or not doc:
                sem_texto += 1
                problemas.append((p.id, antigo, "payload sem texto/document_id"))
                continue
            novo = chunk_id_por_conteudo(doc, texto)
            usos[novo] += 1
            if usos[novo] > 1:                      # colisão: tenta o sufixo do corpus
                candidato = f"{novo}-{usos[novo]}"
                if candidato in validos:
                    novo = candidato
                else:
                    problemas.append((p.id, antigo, f"colisão sem correspondente: {candidato}"))
                    continue
            if novo not in validos:
                fora += 1
                problemas.append((p.id, antigo, f"id calculado não existe no corpus: {novo}"))
                continue
            if novo == antigo:
                ja_ok += 1
                continue
            mudam += 1
            if len(exemplos) < 3:
                exemplos.append((antigo, novo))
            pendentes.append((p.id, novo))
        if prox is None:
            break

    print(f"pontos na coleção: {total}")
    print(f"  já com o id novo      : {ja_ok}")
    print(f"  a atualizar           : {mudam}")
    print(f"  payload incompleto    : {sem_texto}")
    print(f"  id fora do corpus     : {fora}")
    for antigo, novo in exemplos:
        print(f"     {antigo}  →  {novo}")

    if problemas:
        print(f"\n⚠ {len(problemas)} ponto(s) com problema — os 5 primeiros:")
        for pid, antigo, motivo in problemas[:5]:
            print(f"     ponto {pid} ({antigo}): {motivo}")
        print("  Resolva antes de aplicar: id errado em produção é pior que id antigo.")

    if not args.aplicar:
        print("\n(simulação — rode com --aplicar para gravar)")
        return
    if problemas:
        raise SystemExit("\nAbortado: há pontos com problema. Nada foi gravado.")

    print(f"\ngravando {len(pendentes)} atualizações de payload...")
    for i in range(0, len(pendentes), LOTE):
        for pid, novo in pendentes[i:i + LOTE]:
            cliente.set_payload(collection_name=args.colecao,
                                payload={"id_original": novo}, points=[pid])
        print(f"  {min(i + LOTE, len(pendentes))}/{len(pendentes)}")

    # ── verificação: relê e confere ─────────────────────────────────────────
    prox, conferidos, errados = None, 0, 0
    while True:
        pontos, prox = cliente.scroll(collection_name=args.colecao, limit=LOTE,
                                      offset=prox, with_payload=True, with_vectors=False)
        if not pontos:
            break
        for p in pontos:
            carga = p.payload or {}
            conferidos += 1
            if (carga.get("id_original") or "") not in validos:
                errados += 1
        if prox is None:
            break
    print(f"\nverificação: {conferidos} pontos relidos, {errados} com id fora do corpus")
    print("Produção agora devolve os mesmos ids do gabarito." if not errados
          else "⚠ ainda há divergência — investigar antes de usar em avaliação.")


if __name__ == "__main__":
    main()
