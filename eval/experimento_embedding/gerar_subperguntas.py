#!/usr/bin/env python3
"""Decompõe cada comparativa em duas sub-perguntas, uma por lado.

POR QUE SEM OLHAR O GABARITO. Seria fácil passar ao modelo os dois
`source_docs` da pergunta e pedir uma sub-pergunta para cada. Mas produção não
sabe quais documentos a resposta exige — se soubesse, não precisaria buscar. A
decomposição tem de sair do TEXTO DA PERGUNTA, que já nomeia as duas normas
("Como o NIST AI RMF e a Recomendação da UNESCO tratam…"). Usar o gabarito aqui
inflaria o resultado e o revisor perguntaria de onde veio a informação.

REAPROVEITA O CACHE ANTIGO. `eval/results_decomp/subqueries_cache.json` tem 50
pares gerados em 09/2026, indexados pelo TEXTO da pergunta. Os que ainda casam
são reusados — mesma decomposição, menos chamadas, e o experimento fica
comparável ao que foi refutado antes.

Uso:
  python eval/experimento_embedding/gerar_subperguntas.py
  python eval/experimento_embedding/gerar_subperguntas.py --aplicar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ / "eval"))

GABARITO = RAIZ / "eval/data/golden_dezembro.jsonl"
CACHE_ANTIGO = RAIZ / "eval/results_decomp/subqueries_cache.json"
SAIDA = RAIZ / "eval/results/indexes/subperguntas.json"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

# O PROMPT EXIGE FRASE COMPLETA EM PORTUGUÊS. A primeira versão pedia só a
# divisão e o gpt-oss devolvia rótulos telegráficos em inglês ("NIST AI RMF
# transparency treatment"). Isso destruiria a busca: o corpus é bilíngue mas a
# consulta do usuário é sempre em português, e o embedding casa frase com frase,
# não etiqueta com parágrafo. As sub-perguntas do cache de 09/2026 eram frases
# completas, e é com elas que este lote precisa ser comparável.
SISTEMA = (
    """Você divide uma pergunta que compara DUAS normas em duas sub-perguntas de
busca, uma para cada norma.
REGRAS:
1. Escreva em PORTUGUÊS, mesmo que a norma seja estrangeira.
2. Cada sub-pergunta é uma FRASE INTERROGATIVA COMPLETA, não um rótulo.
3. Cada uma nomeia explicitamente a SUA norma e pergunta apenas o que aquela
   norma diz sobre o assunto.
4. Não responda a pergunta; apenas divida.
Responda SOMENTE com JSON válido: {"sub1": "...", "sub2": "..."}
"""
    'Exemplo: pergunta "Como o NIST AI RMF e a Recomendação da UNESCO tratam '
    'transparência?" → {"sub1": "Como o NIST AI RMF trata a transparência em '
    'sistemas de IA?", "sub2": "Como a Recomendação da UNESCO trata a '
    'transparência em sistemas de IA?"}'
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelo", default="gptoss")
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    gold = [json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip()]
    comp = [g for g in gold if g["question_type"] == "comparative"]

    cache = {}
    if CACHE_ANTIGO.exists():
        try:
            cache = json.loads(CACHE_ANTIGO.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    if SAIDA.exists():
        try:
            cache.update(json.loads(SAIDA.read_text(encoding="utf-8")))
        except Exception:
            pass

    faltam = [g for g in comp if g["question"] not in cache]
    print(f"comparativas: {len(comp)}")
    print(f"  já em cache : {len(comp) - len(faltam)}")
    print(f"  a gerar     : {len(faltam)}")

    if not args.aplicar:
        print("\n(relatório apenas — use --aplicar para gerar e gravar)")
        return

    if faltam:
        from lib.common import load_config
        from lib.llm_clients import LLMClient
        cfg = load_config()
        mcfg = next(m for m in cfg["judges"]["models"] if m["name"] == args.modelo)
        # max_tokens folgado e reasoning_effort baixo: gpt-oss e qwen são
        # modelos de raciocínio e gastam o orçamento pensando antes de
        # responder — com 300 tokens a resposta volta truncada, sem JSON.
        cliente = LLMClient(provider=mcfg["provider"], model=mcfg["model"],
                            temperature=0, max_tokens=1200,
                            reasoning_effort="low")
        print(f"\nmodelo: {mcfg['provider']}:{mcfg['model']}\n")

        for i, g in enumerate(faltam, start=1):
            try:
                r = cliente.chat_json(g["question"], system=SISTEMA)
                if isinstance(r, dict) and r.get("sub1") and r.get("sub2"):
                    cache[g["question"]] = {"sub1": str(r["sub1"]).strip(),
                                            "sub2": str(r["sub2"]).strip()}
                else:
                    print(f"  ✗ [{g['qid']}] resposta sem sub1/sub2")
            except Exception as e:
                print(f"  ✗ [{g['qid']}] {type(e).__name__}: {e}")
            if i % 10 == 0 or i == len(faltam):
                SAIDA.parent.mkdir(parents=True, exist_ok=True)
                SAIDA.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
                print(f"  {i}/{len(faltam)}")

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    cobertas = sum(1 for g in comp if g["question"] in cache)
    print(f"\n✓ {SAIDA}")
    print(f"  comparativas com sub-perguntas: {cobertas} de {len(comp)}")

    print("\n── amostra ──")
    for g in comp[:3]:
        s = cache.get(g["question"])
        if not s:
            continue
        print(f"\n[{g['qid']}] {g['question'][:100]}")
        print(f"   sub1: {s['sub1'][:110]}")
        print(f"   sub2: {s['sub2'][:110]}")


if __name__ == "__main__":
    main()
