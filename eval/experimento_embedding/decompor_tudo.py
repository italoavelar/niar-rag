#!/usr/bin/env python3
"""Parte a pergunta em duas SEM SABER o tipo dela — nos três tipos.

O QUE ISTO CORRIGE. A decomposição que funcionou em 29/09 (comparativa 13 → 21
de 75) só era aplicável porque alguém já sabia que a pergunta era comparativa: o
prompt dizia "esta pergunta compara DUAS normas, divida uma por norma". Produção
não sabe. Enquanto a decisão de decompor vier do rótulo do gabarito, o ganho é
truque de oráculo, não método — e um revisor do JBCS diz isso na primeira volta.

A VERSÃO HONESTA, que é esta: o modelo recebe SÓ o texto da pergunta e decide
sozinho se ela pede uma informação ou duas. Roda igual nas 75 factuais, nas 75
multi-hop e nas 75 comparativas, sem nunca ver o tipo nem o gabarito.

ISSO ENTREGA DE GRAÇA UMA SEGUNDA MEDIÇÃO, que talvez valha mais que a primeira:
o campo `partes` é um classificador de tipo derivado só do enunciado. Comparado
ao número de citações que o gabarito exige, ele diz se a decisão "decompor ou
não" é inferível da pergunta. Se for, a objeção "produção não sabe o tipo"
deixa de existir. Se não for, sabemos o custo exato de errar.

POR QUE NÃO REAPROVEITAR O CACHE DAS COMPARATIVAS. Existem 50 pares em
`eval/results_decomp/subqueries_cache.json`, gerados com o prompt que ANUNCIAVA
a comparação. Reusá-los faria um terço do experimento não ser cego. Eles ficam
onde estão, e servem de controle: se a decomposição cega empatar com a
anunciada, é a cega que vai para o artigo.

O EXEMPLO ATÔMICO NO PROMPT NÃO É ENFEITE. Sem ele o modelo divide tudo —
qualquer pergunta pode ser partida em duas se o modelo achar que deve. É a
demonstração de que ele PODE dizer "uma parte só" que dá sentido ao número.

Uso:
  python eval/experimento_embedding/decompor_tudo.py
  python eval/experimento_embedding/decompor_tudo.py --aplicar
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ / "eval"))

GABARITO = RAIZ / "eval/data/golden_dezembro.jsonl"
SAIDA = RAIZ / "eval/results/indexes/decomposicao_cega.json"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

SISTEMA = """Você prepara consultas de busca para um sistema que recupera trechos \
de normas jurídicas brasileiras e internacionais sobre saúde e inteligência \
artificial.

Decida se a pergunta pede UMA informação ou DUAS informações distintas. Duas \
informações distintas são, por exemplo, o que duas normas diferentes dizem sobre \
um assunto, ou dois dispositivos distintos da mesma norma.

Se pedir DUAS, escreva duas sub-perguntas de busca, uma para cada parte.
Se pedir UMA, responda partes=1 e deixe sub1 e sub2 como string vazia.

REGRAS para as sub-perguntas:
1. Escreva em PORTUGUÊS, mesmo que a norma seja estrangeira.
2. Cada sub-pergunta é uma FRASE INTERROGATIVA COMPLETA, nunca um rótulo.
3. Cada uma tem de ser buscável sozinha: repita o nome da norma e o assunto; não \
use "ela", "o primeiro", "esse", "a segunda".
4. Não responda a pergunta; apenas prepare as buscas.

Responda SOMENTE com JSON válido, no formato \
{"partes": 1, "sub1": "", "sub2": ""} ou \
{"partes": 2, "sub1": "...", "sub2": "..."}

Exemplos:

Pergunta: "Como o NIST AI RMF e a Recomendação da UNESCO tratam transparência?"
{"partes": 2, "sub1": "Como o NIST AI RMF trata a transparência em sistemas de \
IA?", "sub2": "Como a Recomendação da UNESCO trata a transparência em sistemas \
de IA?"}

Pergunta: "Que direito de portabilidade a LGPD assegura ao titular e o que o \
controlador precisa informar ao comunicar um incidente de segurança?"
{"partes": 2, "sub1": "Que direito de portabilidade dos dados a LGPD assegura ao \
titular?", "sub2": "O que a LGPD exige que seja informado na comunicação de um \
incidente de segurança?"}

Pergunta: "Qual é o prazo de guarda do prontuário médico segundo o CFM?"
{"partes": 1, "sub1": "", "sub2": ""}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelo", default="gptoss")
    ap.add_argument("--gabarito", default=str(GABARITO))
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    gold = [json.loads(l) for l in Path(args.gabarito).open(encoding="utf-8")
            if l.strip()]
    alvo = [g for g in gold if g.get("qrels_grupos")]

    cache: dict[str, dict] = {}
    if SAIDA.exists():
        try:
            cache = json.loads(SAIDA.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    faltam = [g for g in alvo if g["question"] not in cache]
    print(f"respondíveis no gabarito: {len(alvo)}")
    print(f"  por tipo   : {dict(Counter(g['question_type'] for g in alvo))}")
    print(f"  já em cache: {len(alvo) - len(faltam)}")
    print(f"  a gerar    : {len(faltam)}")

    if faltam and not args.aplicar:
        print("\n(relatório apenas — use --aplicar para chamar o LLM e gravar)")
        return

    if faltam:
        from lib.common import load_config
        from lib.llm_clients import LLMClient
        cfg = load_config()
        mcfg = next(m for m in cfg["judges"]["models"] if m["name"] == args.modelo)
        # max_tokens folgado e reasoning_effort baixo: gpt-oss é modelo de
        # raciocínio e gasta orçamento pensando antes de responder — com 300
        # tokens a resposta volta truncada, sem JSON.
        cliente = LLMClient(provider=mcfg["provider"], model=mcfg["model"],
                            temperature=0, max_tokens=1200,
                            reasoning_effort="low")
        print(f"\nmodelo: {mcfg['provider']}:{mcfg['model']}")
        print("(o modelo NÃO recebe o tipo da pergunta nem o gabarito)\n")

        falhas = 0
        for i, g in enumerate(faltam, start=1):
            try:
                r = cliente.chat_json(g["question"], system=SISTEMA)
                partes = int(r.get("partes", 0)) if isinstance(r, dict) else 0
                if partes == 1:
                    cache[g["question"]] = {"partes": 1, "sub1": "", "sub2": ""}
                elif partes == 2 and r.get("sub1") and r.get("sub2"):
                    cache[g["question"]] = {"partes": 2,
                                            "sub1": str(r["sub1"]).strip(),
                                            "sub2": str(r["sub2"]).strip()}
                else:
                    falhas += 1
                    print(f"  ✗ [{g['qid']}] resposta fora do formato: "
                          f"{str(r)[:110]}")
            except Exception as e:
                falhas += 1
                print(f"  ✗ [{g['qid']}] {type(e).__name__}: {e}")
            if i % 25 == 0 or i == len(faltam):
                SAIDA.parent.mkdir(parents=True, exist_ok=True)
                SAIDA.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
                print(f"  {i}/{len(faltam)}  (falhas: {falhas})")

        SAIDA.parent.mkdir(parents=True, exist_ok=True)
        SAIDA.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        print(f"\n✓ {SAIDA}")

    # ── o classificador implícito: `partes` contra o que o gabarito exige ────
    print("\n\nO DECOMPOSITOR ACERTA SOZINHO QUANTAS PARTES A PERGUNTA TEM?\n")
    print("Linha = o que o gabarito exige; coluna = o que o modelo decidiu, "
          "vendo só o enunciado.\n")
    matriz: dict[tuple, int] = defaultdict(int)
    por_tipo: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    cobertas = 0
    for g in alvo:
        d = cache.get(g["question"])
        if not d:
            continue
        cobertas += 1
        exige = 1 if len(g["qrels_grupos"]) == 1 else 2
        matriz[(exige, d["partes"])] += 1
        por_tipo[g["question_type"]][d["partes"]] += 1

    print(f"{'gabarito':22s}{'modelo: 1 parte':>17s}{'modelo: 2 partes':>18s}")
    print("─" * 57)
    for exige, rot in ((1, "exige 1 citação"), (2, "exige 2 ou mais")):
        a, b = matriz[(exige, 1)], matriz[(exige, 2)]
        print(f"{rot:22s}{a:17d}{b:18d}")
    acertos = matriz[(1, 1)] + matriz[(2, 2)]
    print(f"\n  concordância: {acertos} de {cobertas} "
          f"({acertos/max(cobertas,1):.1%})")
    print(f"  o modelo divide o que não precisava: {matriz[(1, 2)]}"
          f"   (custo: gasta uma vaga do top-5 com o lado inexistente)")
    print(f"  o modelo NÃO divide o que precisava: {matriz[(2, 1)]}"
          f"   (custo: fica igual à consulta única)")

    print("\n\nO QUE O MODELO DECIDIU, POR TIPO\n")
    print(f"{'tipo':16s}{'n':>5s}{'1 parte':>10s}{'2 partes':>10s}{'% dividida':>12s}")
    print("─" * 53)
    for t in ("factual", "multi_hop", "comparative"):
        d = por_tipo.get(t, {})
        n = sum(d.values())
        if not n:
            continue
        print(f"{t:16s}{n:5d}{d.get(1, 0):10d}{d.get(2, 0):10d}"
              f"{d.get(2, 0)/n:11.0%}")
    print("\n  Esta tabela é o resultado que interessa: se a coluna '2 partes'")
    print("  for alta na comparativa e na multi-hop e BAIXA na factual, o tipo")
    print("  é inferível do enunciado e a decomposição pode ser sempre-ligada.")

    print("\n── amostra, uma de cada tipo ──")
    vistos = set()
    for g in alvo:
        t = g["question_type"]
        d = cache.get(g["question"])
        if not d or t in vistos:
            continue
        vistos.add(t)
        print(f"\n[{g['qid']}] {t} · gabarito exige "
              f"{len(g['qrels_grupos'])} citação(ões) · modelo disse "
              f"{d['partes']} parte(s)")
        print(f"   P   : {g['question'][:120]}")
        if d["partes"] == 2:
            print(f"   sub1: {d['sub1'][:115]}")
            print(f"   sub2: {d['sub2'][:115]}")

    if not args.aplicar:
        print("\n(relatório apenas)")


if __name__ == "__main__":
    main()
