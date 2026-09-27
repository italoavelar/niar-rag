#!/usr/bin/env python3
"""Confere que o acervo NÃO responde às perguntas marcadas como irrespondíveis.

O ERRO QUE ISTO EVITA. Uma pergunta declarada irrespondível que na verdade tem
resposta no acervo é pior que não ter pergunta nenhuma: o gabarito passa a punir
o sistema por acertar. E o erro é fácil de cometer — o acervo tem 54 documentos
e quem escreve a pergunta não lembra de todos.

COMO SE VERIFICA. Não por confiança de quem escreveu: busca-se cada pergunta no
corpus de produção, pelo mesmo caminho que a aplicação usa (gemini-embedding-001
com task_type RETRIEVAL_QUERY, contra a coleção do Qdrant), e mostram-se os
trechos mais próximos para leitura humana.

O QUE O SCORE SIGNIFICA, E O QUE NÃO SIGNIFICA. Similaridade alta não prova que
a pergunta tem resposta — uma pergunta sobre a multa do AI Act recupera trechos
sobre sanções da LGPD com score alto, porque o assunto é vizinho. É justamente
esse o caso difícil que se quer no gabarito: o sistema é atraído, e tem de
recusar assim mesmo. Por isso o script ORDENA por score e pede revisão, mas não
derruba nada sozinho.

O que derruba é a leitura: se o trecho recuperado RESPONDE a pergunta, ela não é
irrespondível e sai do acervo.

Uso:
  python eval/experimento_embedding/validar_irrespondiveis.py
  python eval/experimento_embedding/validar_irrespondiveis.py --top 5 --mostrar 25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PROJECT_ROOT = AQUI.parents[1]
DADOS = AQUI / "dados"
ENTRADA = DADOS / "perguntas_irrespondiveis.json"
SAIDA = DADOS / "irrespondiveis_revisao.json"

sys.path.insert(0, str(PROJECT_ROOT / "eval"))

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top", type=int, default=3, help="trechos por pergunta")
    ap.add_argument("--mostrar", type=int, default=15,
                    help="quantas perguntas imprimir, das de maior score")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    from google import genai
    from google.genai import types
    from qdrant_client import QdrantClient

    perguntas = json.loads(ENTRADA.read_text(encoding="utf-8"))
    cliente = genai.Client(api_key=os.getenv("GOOGLE_GENAI_API_KEY")
                           or os.getenv("GOOGLE_API_KEY"))
    qdrant = QdrantClient(url=os.getenv("QDRANT_URL"),
                          api_key=os.getenv("QDRANT_API_KEY"), timeout=120)
    colecao = os.getenv("QDRANT_GEMINI_COLLECTION", "leme_gemini")

    vetores = []
    for i in range(0, len(perguntas), 20):
        lote = [p["question"] for p in perguntas[i:i + 20]]
        resp = cliente.models.embed_content(
            model="gemini-embedding-001", contents=lote,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))
        vetores.extend(e.values for e in resp.embeddings)

    resultados = []
    for p, v in zip(perguntas, vetores):
        achados = qdrant.query_points(collection_name=colecao, query=v,
                                      limit=args.top).points
        resultados.append({
            "n": p["n"], "question": p["question"],
            "motivo": p["motivo_ausencia"],
            "score_max": achados[0].score if achados else 0.0,
            "trechos": [{"score": round(a.score, 4),
                         "documento": a.payload.get("document_id"),
                         "texto": (a.payload.get("texto") or "")[:300]}
                        for a in achados],
        })

    resultados.sort(key=lambda r: -r["score_max"])
    SAIDA.write_text(json.dumps(resultados, ensure_ascii=False, indent=1),
                     encoding="utf-8")

    scores = [r["score_max"] for r in resultados]
    print(f"perguntas buscadas: {len(resultados)} | coleção: {colecao}")
    print(f"score do 1º trecho: máx {max(scores):.3f}  mediana "
          f"{sorted(scores)[len(scores)//2]:.3f}  mín {min(scores):.3f}")
    print(f"\nAS {args.mostrar} DE MAIOR SCORE — são as que mais atraem o sistema, "
          "e as que precisam de leitura:\n")
    for r in resultados[:args.mostrar]:
        print("=" * 78)
        print(f"{r['n']} [{r['motivo']}] score {r['score_max']:.3f}")
        print(f"  P: {r['question']}")
        for t in r["trechos"][:2]:
            print(f"  [{t['score']:.3f}] {t['documento'][:40]}: {t['texto'][:200]!r}")
    print(f"\n✓ {SAIDA}  (todas as {len(resultados)}, ordenadas por score)")


if __name__ == "__main__":
    main()
