#!/usr/bin/env python3
"""Converte o gabarito piloto para citação literal, e com isso o torna reutilizável.

O DEFEITO DO PILOTO, e não é de qualidade. As 100 perguntas de `golden_qa.jsonl`
passam toda auditoria que se faça nelas: zero expressão meta, no máximo 4 âncoras
de grau 2 por pergunta, nenhuma resposta afirmando quantidade que a evidência não
tenha, e as 25 `unanswerable` são genuinamente irrespondíveis. O problema é onde
elas ancoram: em **id de trecho**.

Id de trecho é hash do conteúdo. Muda o recorte, mudam todos os ids, e o gabarito
tem de ser reancorado — foi o que obrigou a reancorar 256 linhas de qrels a cada
limpeza do corpus. Pior: o piloto **não consegue** comparar configurações de
recorte, porque um id do corpus de 1.200 não existe no corpus de 400. Foi por
isso que o experimento de recorte precisou de um gabarito novo.

O QUE ESTE SCRIPT FAZ. De cada âncora de grau 2 extrai a maior frase do
`qrels_text` e guarda os 160 caracteres centrais dela como citação literal. A
citação é um fato, e fato sobrevive a qualquer recorte — a menos que o recorte o
parta ao meio, que é justamente o defeito a medir.

Por que 160 caracteres: curto o bastante para caber dentro de um trecho pequeno
(a mediana da unidade jurídica no acervo é 304), longo o bastante para ser único.

A SAÍDA TEM O MESMO FORMATO DE `dados/perguntas.json`, então as duas famílias
passam pelo mesmo `analisar_recorte.py`. O que o piloto traz de novo e as 91 não
têm: `question_type`, incluindo as 25 **unanswerable** — sem elas não há como
medir se o sistema recusa quando deve, que é metade do problema num RAG jurídico.

`janela` sai como **1200**, e isso é uma limitação a declarar, não um detalhe.
As perguntas do piloto foram escritas a partir dos TRECHOS do corpus, que têm
1.200 caracteres. Ou seja: as 100 nasceram de um dos tamanhos de recorte em
disputa, e por construção favorecem esse tamanho.

CONSEQUÊNCIA PRÁTICA. O conjunto somado (piloto + as 91) NÃO SERVE para comparar
tamanhos de recorte — metade dele é enviesada para 1.200. As 91 existem
justamente porque o piloto não servia para isso: elas variam a janela em 600,
1400 e 3000 para que o viés vire medida em vez de ficar escondido.

Para o que o conjunto somado SERVE, que é o artigo de dezembro: lá o recorte é
fixo e o que varia é o modelo de embedding. Pergunta escrita de um trecho de
1.200 não favorece Gemini sobre BGE, então o viés não contamina essa comparação.

Este arquivo substitui o `bench_chunking.py`, que media as mesmas coisas com
menos condições, só o BGE e o gabarito antigo.

Uso:
  python eval/experimento_embedding/converter_piloto.py
  python eval/experimento_embedding/converter_piloto.py --aplicar
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import unicodedata
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PROJECT_ROOT = AQUI.parents[1]
GOLDEN = PROJECT_ROOT / "eval/data/golden_qa.jsonl"
DADOS = AQUI / "dados"
SAIDA = DADOS / "perguntas_piloto.json"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

NUCLEO_CHARS = 160
# Frase com pelo menos 60 caracteres úteis. O piso existe para não pegar
# fragmento de cabeçalho ("Art. 5º") como se fosse a evidência.
FRASE = re.compile(r"[^.!?;]{60,}")


def normalizar(t: str) -> str:
    t = (t or "").replace("­", "")
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def sinais(texto: str) -> set[str]:
    """Tokens que servem para reconhecer o assunto, inclusive entre idiomas.

    Palavras de 5+ letras pegam o caso monolíngue. Números e siglas em caixa alta
    atravessam a tradução — a resposta em português cita "UNESCO", "NIST",
    "2338" e "four" exatamente como o trecho em inglês.
    """
    base = ''.join(c for c in unicodedata.normalize("NFKD", texto or "")
                   if not unicodedata.combining(c))
    return ({w.lower() for w in re.findall(r"[A-Za-z]{5,}", base)}
            | set(re.findall(r"\d[\d.,]*", base))
            | {s.lower() for s in re.findall(r"\b[A-Z]{3,}\b", base)})


def nucleo(texto: str, resposta: str) -> str | None:
    """160 caracteres da frase da âncora que MAIS SUSTENTA a resposta.

    POR QUE NÃO A MAIOR FRASE. A primeira versão pegava a mais longa, e 37% das
    citações saíam fora do assunto: para "o que precisa ser garantido quanto ao
    consentimento do paciente", ela devolvia "as empresas de arquivamento de
    dados deverão ter sede estabelecida em território brasileiro" — a frase mais
    comprida da âncora, e nada a ver com a pergunta.

    Citação fora do assunto estraga a medição nos dois sentidos: exige do
    recuperador um texto que não é a evidência, e deixa de exigir o que é.

    A escolha agora é por sobreposição de sinais com a RESPOSTA — é ela que
    afirma o fato, e a evidência é a frase que a sustenta. Comprimento fica só
    como critério de desempate.
    """
    frases = [f.strip() for f in FRASE.findall(texto or "")]
    if not frases:
        return None
    alvo = sinais(resposta)
    melhor = max(frases, key=lambda f: (len(alvo & sinais(f)), len(f)))
    if len(melhor) <= NUCLEO_CHARS:
        return melhor if len(normalizar(melhor)) >= 40 else None
    # Centra a janela na região de maior densidade de sinais, não no meio
    # geométrico: a frase pode ser longa e só um pedaço dela ser a evidência.
    janelas = [(len(alvo & sinais(melhor[i:i + NUCLEO_CHARS])), -i)
               for i in range(0, len(melhor) - NUCLEO_CHARS + 1, 20)]
    inicio = -max(janelas)[1]
    corte = melhor[inicio:inicio + NUCLEO_CHARS].strip()
    return corte if len(normalizar(corte)) >= 40 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aplicar", action="store_true", help="grava o arquivo")
    args = ap.parse_args()

    gold = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]
    corpus = {r["id"]: r for r in (json.loads(l) for l in
              (PROJECT_ROOT / "data/processed/documents.jsonl").open(encoding="utf-8"))}
    doc_inteiro: dict[str, str] = {}
    for r in corpus.values():
        d = r["metadata"]["document_id"]
        doc_inteiro[d] = doc_inteiro.get(d, "") + " " + normalizar(r["text"])

    saida, perdidas, sem_frase = [], [], []
    for g in gold:
        tipo = g["question_type"]
        qrels = g.get("qrels") or {}
        textos = g.get("qrels_text") or {}

        if tipo == "unanswerable":
            # Sem evidência por construção: a resposta correta é a recusa.
            saida.append({
                "n": g["qid"], "question": g["question"],
                "answer": g["reference_answer"], "evidencia": [], "n_evidencias": 0,
                "documento": None, "forma": None, "janela": 1200,
                "passagem_texto": None, "question_type": tipo,
                "difficulty": g.get("difficulty"), "theme": g.get("theme"),
                "source_lang": g.get("source_lang"), "origem": "piloto",
            })
            continue

        ev, docs = [], []
        for cid, grau in qrels.items():
            if int(grau) < 2:
                continue
            n = nucleo(textos.get(cid, ""), g["reference_answer"])
            if n is None:
                sem_frase.append((g["qid"], cid))
                continue
            doc = cid.rsplit("_", 1)[0]
            # A CITAÇÃO TEM DE ESTAR NO CORPUS DE HOJE. O `qrels_text` é uma foto
            # do dia do julgamento e pode carregar ruído que a limpeza removeu —
            # guardar uma citação que já não existe criaria gabarito cobrando
            # texto inexistente, e toda condição erraria igual.
            if normalizar(n) not in doc_inteiro.get(doc, ""):
                perdidas.append((g["qid"], cid, n[:60]))
                continue
            ev.append(n)
            docs.append(doc)

        if not ev:
            perdidas.append((g["qid"], "—", "nenhuma citação sobreviveu"))
            continue

        saida.append({
            "n": g["qid"], "question": g["question"],
            "answer": g["reference_answer"], "evidencia": ev, "n_evidencias": len(ev),
            "documento": docs[0] if len(set(docs)) == 1 else None,
            # Quando a pergunta cruza documentos, `documento` fica nulo e a
            # procedência sobraria só dentro do texto das citações. Guardar a
            # lista aqui é o que permite conferir se a busca trouxe OS DOIS
            # lados — mesma forma que `gerar_comparativas.py` usa.
            "documentos": sorted(set(docs)) if len(set(docs)) > 1 else None,
            "forma": None, "janela": 1200, "passagem_texto": None,
            "question_type": tipo, "difficulty": g.get("difficulty"),
            "theme": g.get("theme"), "source_lang": g.get("source_lang"),
            "origem": "piloto",
        })

    resp = [r for r in saida if r["question_type"] != "unanswerable"]
    print(f"gabarito piloto: {len(gold)} perguntas")
    print(f"  convertidas          : {len(saida)}")
    print(f"    com citação literal: {len(resp)}")
    print(f"    unanswerable       : {len(saida) - len(resp)}")
    print(f"  perdidas             : {len({p[0] for p in perdidas})} pergunta(s)")
    for q, c, t in perdidas[:6]:
        print(f"      {q} {c[:44]}: {t!r}")
    if sem_frase:
        print(f"  âncora sem frase de 60+ chars: {len(sem_frase)} → {sem_frase[:3]}")
    if resp:
        print(f"\n  por tipo       : {dict(Counter(r['question_type'] for r in saida))}")
        print(f"  por dificuldade: {dict(Counter(r['difficulty'] for r in saida))}")
        print(f"  citações por pergunta: mediana "
              f"{st.median([r['n_evidencias'] for r in resp]):.0f}, "
              f"máx {max(r['n_evidencias'] for r in resp)}")
        print(f"  documento único: {sum(1 for r in resp if r['documento'])}/{len(resp)}"
              "  (o resto cruza documentos — é o caso das comparativas)")

    if args.aplicar:
        SAIDA.write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n✓ {SAIDA}")
        print("  Mesmo formato de perguntas.json — passa pelo mesmo analisar_recorte.py.")
    else:
        print("\n(relatório apenas — use --aplicar para gravar)")


if __name__ == "__main__":
    main()
