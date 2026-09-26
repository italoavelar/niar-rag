#!/usr/bin/env python3
"""Mede qual recorte recupera melhor a evidência — as 4 condições x 2 modelos.

A MÉTRICA É DE COMPLETUDE, e isso não é detalhe. Metade das perguntas tem duas
citações, e as duas precisam estar no que voltou para a resposta ser possível.
Contar "achou pelo menos um trecho relevante" mede outra coisa e mede para cima:
no gabarito antigo essa diferença levou multi-hop de 84% para 12%. Aqui as duas
são reportadas lado a lado justamente para a distância ficar visível.

    Recall@k     ao menos UMA citação da pergunta aparece no top-k
    Junta@k      TODAS as citações aparecem no top-k          <- a que vale

A UNIDADE DE RECUPERAÇÃO MUDA POR CONDIÇÃO. Em T400/T1200/T3500 o que é embutido
e o que é devolvido são a mesma coisa: o trecho. Em parent-child não — embute-se
o filho (400) e devolve-se o pai (3500). Comparar top-k de filho com top-k de
trecho seria comparar 400 caracteres com 3.500. Então o k conta PAIS DISTINTOS,
descidos pelo ranking dos filhos: é o que o leitor receberia.

O VIÉS DE JANELA VIRA MEDIDA. Cada pergunta carrega a janela de texto de onde
saiu (600, 1400 ou 3000). Se a vantagem de um recorte se concentrar nas
perguntas de janela parecida, está declarado no recorte por janela. Se não se
concentrar, o resultado é robusto. Ver `sortear_passagens.py`.

Uso:
  python eval/experimento_embedding/analisar_recorte.py
  python eval/experimento_embedding/analisar_recorte.py --modelo bge
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
PROJECT_ROOT = AQUI.parents[1]
DADOS = AQUI / "dados"
RESULTADOS = AQUI / "resultados"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

# PC1200 separa a ESTRATEGIA parent-child da UNIDADE DE SAIDA. O PC original e
# filho 400 / pai 3500, o que mistura as duas: quando ele perde num orcamento
# apertado, pode ser a estrategia que nao funciona OU so o pai de 3500 que nao
# cabe. No PC1200 os pais sao exatamente os trechos do T1200, entao a unica
# diferenca entre as duas condicoes e POR ONDE SE BUSCA: pelo filho de 400 ou
# pelo proprio trecho de 1200. E a comparacao que isola a estrategia.
CONDICOES = ("T400", "T1200", "T3500", "PC", "PC1200")
KS = (1, 3, 5, 10)

# ORÇAMENTO EM CARACTERES, e por que ele é a comparação que vale.
#
# Em k fixo o recorte grande ganha de graça: no top-5, T400 devolve ~1.600
# caracteres e T3500 devolve ~14.750. Quem tem nove vezes mais texto acerta mais
# citação sem recuperar melhor nada — mede-se o tamanho da resposta, não a
# qualidade da busca. A ordem sai monotônica no tamanho do trecho, que é
# exatamente o sintoma de um teste medindo a própria variável independente.
#
# O que a aplicação de verdade tem é orçamento de contexto, não número de
# trechos. Fixando os caracteres, cada condição gasta o mesmo e a pergunta vira
# a certa: com este tanto de texto para gastar, qual recorte entrega a evidência
# completa? Aí trecho pequeno tem a chance de ganhar por precisão — cabem mais
# deles no mesmo orçamento.
ORCAMENTOS = (2000, 5000, 10000)
MODELOS = {
    "bge": {"prefixo": "emb_", "dim": 1024},
    "gemini": {"prefixo": "emb_gemini_", "dim": 3072},
}


def normalizar(t: str) -> str:
    """Mesma normalização do validador: conteúdo, não bytes."""
    t = (t or "").replace("­", "")
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def carregar_npz(caminho: Path) -> dict:
    """Materializa os arrays de uma vez.

    `np.load` devolve um NpzFile preguiçoso: CADA acesso a `d["matrix"]`
    descomprime o array inteiro outra vez. Indexar dentro de um laço custou uma
    hora de CPU antes de alguém notar. Aqui se lê uma vez e pronto.
    """
    with np.load(caminho, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def embutir_consultas(modelo: str, perguntas: list[str]) -> np.ndarray:
    if modelo == "bge":
        from sentence_transformers import SentenceTransformer
        print("  carregando BGE-m3 (CPU) para as consultas...", flush=True)
        m = SentenceTransformer("BAAI/bge-m3", device="cpu")
        return m.encode(perguntas, normalize_embeddings=True,
                        batch_size=8, show_progress_bar=False).astype("float32")

    from google import genai
    from google.genai import types
    from lib.common import get_env
    cliente = genai.Client(api_key=get_env("GOOGLE_GENAI_API_KEY", required=True))
    saida = []
    for i in range(0, len(perguntas), 20):
        resp = cliente.models.embed_content(
            model="gemini-embedding-001", contents=perguntas[i:i + 20],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))
        saida.extend(e.values for e in resp.embeddings)
    v = np.asarray(saida, dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def textos_recuperados(cond: str, ordem_linha: np.ndarray, k: int,
                       textos: list[str], pai_de: list[str] | None,
                       texto_do_pai: dict[str, str]) -> str:
    """O que o leitor receberia no top-k, já concatenado e normalizado."""
    if pai_de is None:
        return " ".join(textos[j] for j in ordem_linha[:k])
    vistos, juntos = [], []
    for j in ordem_linha:
        p = pai_de[j]
        if p in vistos:
            continue
        vistos.append(p)
        juntos.append(texto_do_pai[p])
        if len(vistos) == k:
            break
    return " ".join(juntos)


def por_orcamento(ordem_linha: np.ndarray, orcamento: int, textos: list[str],
                  pai_de: list[str] | None,
                  texto_do_pai: dict[str, str]) -> tuple[str, int]:
    """Desce o ranking até gastar o orçamento. Devolve (texto, nº de unidades).

    A unidade que entra inteira é a que o leitor receberia: o trecho, ou o pai
    no parent-child. A última unidade entra mesmo estourando um pouco — cortá-la
    ao meio criaria dentro do teste exatamente o defeito que o teste mede.
    """
    juntos, vistos, gasto = [], [], 0
    for j in ordem_linha:
        if pai_de is None:
            t = textos[j]
        else:
            p = pai_de[j]
            if p in vistos:
                continue
            vistos.append(p)
            t = texto_do_pai[p]
        juntos.append(t)
        gasto += len(t)
        if gasto >= orcamento:
            break
    return " ".join(juntos), len(juntos)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelo", choices=list(MODELOS), help="só um modelo")
    args = ap.parse_args()

    perguntas = json.loads((DADOS / "perguntas.json").read_text(encoding="utf-8"))
    pais_texto = {}
    for arq in ("PC_pais.json", "PC1200_pais.json"):
        caminho = DADOS / arq
        if caminho.exists():
            pais_texto.update({k: normalizar(v) for k, v in
                               json.loads(caminho.read_text(encoding="utf-8")).items()})
    evidencias = [[normalizar(e) for e in p["evidencia"]] for p in perguntas]

    print(f"perguntas: {len(perguntas)} | citações: {sum(len(e) for e in evidencias)}")
    print(f"  por janela : {dict(sorted(Counter(p['janela'] for p in perguntas).items()))}")
    print(f"  por nº de evidências: "
          f"{dict(sorted(Counter(p['n_evidencias'] for p in perguntas).items()))}")

    linhas: list[dict] = []
    for modelo in ([args.modelo] if args.modelo else list(MODELOS)):
        cfg = MODELOS[modelo]
        print(f"\n{'='*78}\n{modelo.upper()}\n{'='*78}")
        Q = embutir_consultas(modelo, [p["question"] for p in perguntas])

        for cond in CONDICOES:
            arq = RESULTADOS / f"{cfg['prefixo']}{cond}.npz"
            if not arq.exists():
                print(f"  {cond}: falta {arq.name}")
                continue
            d = carregar_npz(arq)
            M = d["matrix"]
            if M.shape[1] != cfg["dim"]:
                print(f"  {cond}: dimensão {M.shape[1]} ≠ {cfg['dim']} — pulando")
                continue

            regs = [json.loads(l) for l in (DADOS / f"{cond}.jsonl")
                    .open(encoding="utf-8") if l.strip()]
            if len(regs) != M.shape[0]:
                print(f"  {cond}: {M.shape[0]} vetores para {len(regs)} trechos — pulando")
                continue

            textos = [normalizar(r["text"]) for r in regs]
            pai_de = ([r["metadata"]["pai_id"] for r in regs]
                      if cond.startswith("PC") else None)
            ordem = np.argsort(-(Q @ M.T), axis=1)

            for k in KS:
                junta = recall = 0
                por_janela: dict[int, list[int]] = {}
                for i, evs in enumerate(evidencias):
                    visto = textos_recuperados(cond, ordem[i], k, textos,
                                               pai_de, pais_texto)
                    achou = sum(1 for e in evs if e in visto)
                    completo = int(achou == len(evs))
                    junta += completo
                    recall += int(achou > 0)
                    por_janela.setdefault(perguntas[i]["janela"], []).append(completo)
                linhas.append({
                    "modelo": modelo, "condicao": cond, "k": k,
                    "junta": junta / len(perguntas),
                    "recall": recall / len(perguntas),
                    "por_janela": {j: sum(v) / len(v) for j, v in sorted(por_janela.items())},
                })

            # TETO DO RECORTE: quantas perguntas são sequer POSSÍVEIS nele.
            # Se a fronteira do corte parte uma citação ao meio, nenhuma busca
            # recupera aquela evidência inteira — o trecho que a contém não
            # existe. É defeito do recorte e conta como perda, mas separar o
            # teto do desempenho diz QUANTO da diferença é corte e quanto é
            # busca.
            universo = (list(dict.fromkeys(pais_texto[p] for p in pai_de))
                        if pai_de else textos)
            possiveis = [int(all(any(e in t for t in universo) for e in evs))
                         for evs in evidencias]
            teto = sum(possiveis) / len(possiveis)

            for orc in ORCAMENTOS:
                junta = recall = 0
                unidades, gastos, acertos = [], [], []
                por_janela: dict[int, list[int]] = {}
                for i, evs in enumerate(evidencias):
                    visto, n = por_orcamento(ordem[i], orc, textos, pai_de, pais_texto)
                    unidades.append(n)
                    gastos.append(len(visto))
                    achou = sum(1 for e in evs if e in visto)
                    completo = int(achou == len(evs))
                    acertos.append(completo)
                    junta += completo
                    recall += int(achou > 0)
                    por_janela.setdefault(perguntas[i]["janela"], []).append(completo)
                linhas.append({
                    "modelo": modelo, "condicao": cond, "orcamento": orc,
                    "junta": junta / len(perguntas),
                    "recall": recall / len(perguntas),
                    "unidades": float(np.mean(unidades)),
                    "chars_gastos": float(np.mean(gastos)),
                    "teto": teto,
                    # por pergunta, para o teste pareado: as mesmas 40 perguntas
                    # passam por todas as condições, então comparar proporções
                    # independentes jogaria fora a informação do pareamento.
                    "acertos": acertos,
                    "por_janela": {j: sum(v) / len(v) for j, v in sorted(por_janela.items())},
                })

            print(f"\n  {cond}  ({M.shape[0]} vetores"
                  f"{f', {len(set(pai_de))} pais' if pai_de else ''})")
            print(f"    {'k':>3s} {'Junta@k':>9s} {'Recall@k':>9s}   "
                  f"{'j600':>6s} {'j1400':>6s} {'j3000':>6s}")
            for r in [x for x in linhas
                      if x["modelo"] == modelo and x["condicao"] == cond and "k" in x]:
                pj = r["por_janela"]
                print(f"    {r['k']:3d} {r['junta']:8.0%} {r['recall']:9.0%}   "
                      + " ".join(f"{pj.get(j, float('nan')):6.0%}" for j in (600, 1400, 3000)))
            print(f"    {'orç.':>6s} {'Junta':>7s} {'Recall':>7s} {'unid.':>7s}")
            for r in [x for x in linhas
                      if x["modelo"] == modelo and x["condicao"] == cond and "orcamento" in x]:
                print(f"    {r['orcamento']:6d} {r['junta']:6.0%} {r['recall']:7.0%} "
                      f"{r['unidades']:7.1f}")

    # ── quadro comparativo ──────────────────────────────────────────────────
    print(f"\n{'='*78}\nJunta@k — k fixo. CONFUNDIDO: k igual, texto devolvido diferente.\n"
          f"{'='*78}")
    for modelo in sorted({r["modelo"] for r in linhas}):
        print(f"\n{modelo.upper():8s} " + "".join(f"{'k='+str(k):>10s}" for k in KS))
        for cond in CONDICOES:
            vals = {r["k"]: r["junta"] for r in linhas if "k" in r
                    and r["modelo"] == modelo and r["condicao"] == cond}
            if vals:
                print(f"  {cond:6s} " + "".join(f"{vals.get(k, 0):9.0%} " for k in KS))

    print(f"\n{'='*78}\nJunta por ORÇAMENTO de caracteres — é esta a comparação justa\n"
          f"{'='*78}")
    for modelo in sorted({r["modelo"] for r in linhas}):
        print(f"\n{modelo.upper():8s} " + "".join(f"{str(o)+' ch':>11s}" for o in ORCAMENTOS))
        for cond in CONDICOES:
            vals = {r["orcamento"]: r for r in linhas if "orcamento" in r
                    and r["modelo"] == modelo and r["condicao"] == cond}
            if vals:
                print(f"  {cond:6s} " + "".join(
                    f"{vals[o]['junta']:7.0%} ({vals[o]['unidades']:.0f}) " if o in vals
                    else f"{'—':>11s}" for o in ORCAMENTOS))
    print("\n  (entre parênteses: quantas unidades cabem no orçamento, em média)")

    # ── o orçamento foi mesmo igual? ────────────────────────────────────────
    print(f"\n{'='*78}\nQuanto texto cada condição GASTOU de verdade\n{'='*78}")
    print("A última unidade entra inteira mesmo estourando o teto — cortá-la ao meio\n"
          "criaria dentro do teste o defeito que o teste mede. Com unidade grande o\n"
          "estouro é grande, então o orçamento nominal não é o gasto real.\n")
    print(f"  {'cond':6s} {'teto':>6s} " + "".join(f"{str(o):>12s}" for o in ORCAMENTOS))
    for cond in CONDICOES:
        rs = {r["orcamento"]: r for r in linhas if "orcamento" in r
              and r["modelo"] == list(MODELOS)[0] and r["condicao"] == cond}
        if rs:
            print(f"  {cond:6s} {rs[ORCAMENTOS[0]]['teto']:5.0%} "
                  + "".join(f"{rs[o]['chars_gastos']:11.0f} " for o in ORCAMENTOS))

    # ── significância pareada (McNemar exato) ───────────────────────────────
    print(f"\n{'='*78}\nAs diferenças resistem a n=40?  (McNemar pareado, exato)\n{'='*78}")
    print("Uma pergunta vale 2,5 pontos percentuais. Diferença de 5 pontos são DUAS\n"
          "perguntas — pode ser sorte. O teste é pareado porque são as MESMAS 40\n"
          "perguntas em todas as condições.\n")
    from math import comb

    def mcnemar(a: list[int], b: list[int]) -> tuple[int, int, float]:
        """b ganha de a em `vence`, perde em `perde`. p bicaudal exato."""
        vence = sum(1 for x, y in zip(a, b) if y > x)
        perde = sum(1 for x, y in zip(a, b) if y < x)
        n = vence + perde
        if n == 0:
            return 0, 0, 1.0
        k = min(vence, perde)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
        return vence, perde, p

    familia_recorte: list[tuple[str, float]] = []
    familia_modelo: list[tuple[str, float]] = []

    for modelo in sorted({r["modelo"] for r in linhas}):
        for orc in ORCAMENTOS:
            rs = {r["condicao"]: r for r in linhas if r.get("orcamento") == orc
                  and r["modelo"] == modelo}
            if len(rs) < 2:
                continue
            print(f"\n  {modelo.upper()} @ {orc} chars")
            pares = [("T1200", "T400"), ("T3500", "T1200"), ("PC", "T1200"),
                     ("PC", "T3500"), ("PC1200", "T1200"), ("PC1200", "PC")]
            for x, y in pares:
                if x not in rs or y not in rs:
                    continue
                v, d, p = mcnemar(rs[y]["acertos"], rs[x]["acertos"])
                familia_recorte.append((f"{modelo} @{orc} {x} vs {y}", p))
                sinal = "significativo" if p < 0.05 else "NÃO significativo"
                print(f"    {x:6s} vs {y:6s}  {rs[x]['junta']:4.0%} vs {rs[y]['junta']:4.0%}  "
                      f"| {x} ganha {v}, perde {d}  | p={p:.3f}  {sinal}")

    # BGE contra Gemini, na mesma condição
    print(f"\n  GEMINI vs BGE, mesma condição e mesmo orçamento")
    for orc in ORCAMENTOS:
        for cond in CONDICOES:
            a = [r for r in linhas if r.get("orcamento") == orc
                 and r["condicao"] == cond and r["modelo"] == "bge"]
            b = [r for r in linhas if r.get("orcamento") == orc
                 and r["condicao"] == cond and r["modelo"] == "gemini"]
            if not (a and b):
                continue
            v, d, p = mcnemar(a[0]["acertos"], b[0]["acertos"])
            familia_modelo.append((f"@{orc} {cond}", p))
            sinal = "significativo" if p < 0.05 else "não significativo"
            print(f"    {cond:6s} @{orc:6d}  {b[0]['junta']:4.0%} vs {a[0]['junta']:4.0%}  "
                  f"| ganha {v}, perde {d}  | p={p:.3f}  {sinal}")

    # ── correção para testes múltiplos ──────────────────────────────────────
    #
    # POR QUE NÃO BONFERRONI. Bonferroni controla a chance de UM falso positivo
    # em toda a família, e para isso assume os testes independentes. Aqui eles
    # não são: as mesmas 91 perguntas passam por todas as condições, e os três
    # orçamentos de uma mesma condição medem quase a mesma coisa. Com testes
    # correlacionados a correção fica conservadora demais e enterra efeito real.
    #
    # Benjamini-Hochberg controla a FRAÇÃO ESPERADA DE FALSOS entre os
    # resultados declarados (FDR). É garantia mais fraca que a de Bonferroni —
    # aceita que alguns dos declarados estejam errados, contanto que a proporção
    # fique sob controle — e é a escolha usual para uma bateria de comparações
    # correlacionadas como esta. As duas são reportadas, e a mais severa
    # (Bonferroni) fica visível para quem quiser o critério duro.
    #
    # AS FAMÍLIAS SÃO SEPARADAS porque respondem a perguntas diferentes: uma é
    # "qual recorte", a outra é "qual modelo". Corrigir juntas penalizaria uma
    # pela quantidade de testes da outra.
    def benjamini_hochberg(testes: list[tuple[str, float]], alfa: float = 0.05):
        ordenados = sorted(testes, key=lambda t: t[1])
        m = len(ordenados)
        corte = 0
        for i, (_, p) in enumerate(ordenados, start=1):
            if p <= i / m * alfa:
                corte = i
        return ordenados, corte, m

    print(f"\n{'='*78}\nCorreção para testes múltiplos\n{'='*78}")
    for nome, familia in (("RECORTE", familia_recorte), ("MODELO", familia_modelo)):
        if not familia:
            continue
        ordenados, corte, m = benjamini_hochberg(familia)
        bonf = 0.05 / m
        print(f"\n  família {nome}: {m} testes | Bonferroni exige p<{bonf:.4f} | "
              f"BH declara os {corte} menores")
        for i, (rotulo, p) in enumerate(ordenados, start=1):
            if p >= 0.2:
                break
            marca = ("BH+Bonf" if (i <= corte and p < bonf)
                     else "BH     " if i <= corte else "       ")
            print(f"    {marca}  p={p:.4f}  (limiar BH {i/m*0.05:.4f})  {rotulo}")

    destino = RESULTADOS / "recorte_resultados.json"
    destino.write_text(json.dumps(linhas, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✓ {destino}")


if __name__ == "__main__":
    main()
