#!/usr/bin/env python3
"""Embute as quatro condições de recorte com o Gemini, em .npz.

A PONTE QUE FALTAVA. O `colab_embutir.py` resolve o BGE-m3, que é modelo local e
precisa de GPU. O Gemini é chamada de API e roda aqui — mas o
`src/build_vectorstore.py` escreve no Qdrant de produção, não em arquivo, e o
`bench_chunking.py` lê `.npz`. Este script fecha o vão, e só isso: não toca em
produção, não toca no Qdrant, só grava arquivo em `resultados/`.

O QUE ELE FAZ DE DIFERENTE

  1. Embute TEXTO DISTINTO, não trecho. As quatro condições somam 26.586
     trechos, mas só 15.872 textos de embedding distintos — T400 e PC
     compartilham muito filho idêntico. A chave é o `embedding_text_hash`
     (hash do texto, sem o id), que existe exatamente para isso. São 40% de
     chamada a menos e 1,1 M de token a menos. Medido em 22/set.

  2. Retoma de onde parou. Cada vetor vai para `cache_gemini.jsonl` assim que
     chega. Queda de rede, 429 que não cede, Ctrl+C no meio: roda de novo e ele
     pula o que já tem. Nada se perde e nada se paga duas vezes.

  3. Acha o próprio ritmo. O limite depende do plano, e chutar número de
     documentação envelhece mal. Em vez disso: a cada 429 a pausa dobra e o
     lote é repetido; a cada 20 lotes seguidos sem erro ela cede 20% de volta,
     até o piso. Em plano pago ela desce sozinha para o piso e fica lá.

TASK TYPE E TÍTULO seguem a produção (`src/build_vectorstore.py:188`):
RETRIEVAL_DOCUMENT com o mesmo `title`. Não é detalhe cosmético — o vetor muda
com eles, e o objetivo do experimento é escolher o recorte que o LEME vai servir
de verdade. Medir com uma configuração e servir outra responderia à pergunta
errada. As quatro condições recebem tratamento idêntico, então a comparação
entre elas fica justa de qualquer jeito.

Uso:
  python eval/experimento_embedding/gemini_embutir.py            # as quatro
  python eval/experimento_embedding/gemini_embutir.py --estimar  # só o custo
  python eval/experimento_embedding/gemini_embutir.py --so T400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

from embedding_text import (EMBEDDING_TEXT_PROFILE, build_embedding_text,  # noqa: E402
                            embedding_text_hash)

# PC1200 e a variante que separa a ESTRATEGIA parent-child da UNIDADE DE SAIDA:
# filho 400 / pai 1200, onde os pais sao exatamente os trechos do T1200.
CONDICOES = ("T3500", "T1200", "T400", "PC", "PC1200")
MODELO = "gemini-embedding-001"
DIM = 3072
TITULO = "Base de Conhecimento NIAR Saúde"   # igual ao de produção
# O cache de retomada é BINÁRIO, em dois arquivos que crescem juntos: os bytes
# dos vetores num, os hashes no outro, na mesma ordem. A primeira versão gravava
# JSON e daria 556 MB para os 15.872 vetores — 3072 floats viram ~34 KB de texto
# cada. Em float32 cru são 195 MB, e a leitura é uma chamada de `np.fromfile` em
# vez de 15.872 `json.loads`.
CACHE_VETORES = RESULTADOS / "cache_gemini.f32"
CACHE_HASHES = RESULTADOS / "cache_gemini.hashes"

# 1 token ≈ 4,46 caracteres, medido neste corpus. Serve só para estimar custo.
CHARS_POR_TOKEN = 4.46


def carregar_condicao(nome: str) -> list[dict]:
    caminho = DADOS / f"{nome}.jsonl"
    if not caminho.exists():
        return []
    with caminho.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def carregar_cache() -> dict[str, np.ndarray]:
    """Vetores já calculados, por hash do texto.

    Os dois arquivos podem discordar se a execução morreu entre gravar um e o
    outro. Nesse caso vale o menor dos dois: um vetor a mais é recalculado, que
    é barato; um hash apontando para bytes que não existem seria vetor errado
    entrando no `.npz` sem ninguém perceber.
    """
    if not (CACHE_VETORES.exists() and CACHE_HASHES.exists()):
        return {}
    hashes = CACHE_HASHES.read_text(encoding="utf-8").split()
    dados = np.fromfile(CACHE_VETORES, dtype="float32")
    n = min(len(hashes), len(dados) // DIM)
    if n < len(hashes) or n < len(dados) // DIM:
        print(f"  (cache truncado por queda: {len(hashes)} hashes, "
              f"{len(dados)//DIM} vetores — usando {n})")
    if n == 0:
        return {}
    matriz = dados[:n * DIM].reshape(n, DIM)
    return {h: matriz[i] for i, h in enumerate(hashes[:n])}


def normalizar(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n).astype("float32")


class Ritmo:
    """Pausa entre lotes que se ajusta sozinha ao limite do plano.

    Não há número certo para escrever aqui: o limite muda com o plano e com o
    que a Google publica. O que funciona em qualquer um é reagir — subir quando
    leva 429, descer quando não leva.
    """

    def __init__(self, piso: float, teto: float = 60.0):
        self.piso = piso
        self.teto = teto
        self.atual = piso
        self.seguidos = 0
        self.total_429 = 0

    def ok(self) -> None:
        self.seguidos += 1
        if self.seguidos >= 20 and self.atual > self.piso:
            self.atual = max(self.piso, self.atual * 0.8)
            self.seguidos = 0

    def bateu_no_limite(self) -> float:
        self.total_429 += 1
        self.seguidos = 0
        self.atual = min(self.teto, max(self.piso, self.atual * 2) if self.atual else 1.0)
        return self.atual


def eh_limite(erro: Exception) -> bool:
    texto = str(erro).lower()
    return "429" in texto or "resource_exhausted" in texto or "rate limit" in texto


def embutir(textos: list[str], lote: int, ritmo: Ritmo, tentativas: int):
    """Gera vetores lote a lote, devolvendo (índice_inicial, vetores) a cada lote."""
    from google import genai
    from google.genai import types

    sys.path.insert(0, str(PROJECT_ROOT / "eval"))
    from lib.common import get_env

    cliente = genai.Client(api_key=get_env("GOOGLE_GENAI_API_KEY", required=True))
    config = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", title=TITULO)

    i = 0
    while i < len(textos):
        pedaco = textos[i:i + lote]
        for tentativa in range(1, tentativas + 1):
            try:
                resp = cliente.models.embed_content(
                    model=MODELO, contents=pedaco, config=config)
                vetores = [np.asarray(e.values, dtype="float32") for e in resp.embeddings]
                if len(vetores) != len(pedaco):
                    raise RuntimeError(
                        f"a API devolveu {len(vetores)} vetores para {len(pedaco)} textos")
                ritmo.ok()
                yield i, vetores
                break
            except Exception as erro:
                if eh_limite(erro) and tentativa < tentativas:
                    espera = ritmo.bateu_no_limite()
                    print(f"\n  429 — pausa para {espera:.1f}s e repete o lote "
                          f"(tentativa {tentativa}/{tentativas})", flush=True)
                    time.sleep(espera)
                    continue
                # Erro que não é limite, ou tentativas esgotadas: para. O que já
                # foi calculado está no cache e a próxima execução retoma dali.
                raise
        i += lote
        if ritmo.atual and i < len(textos):
            time.sleep(ritmo.atual)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--so", choices=CONDICOES, help="uma condição só")
    ap.add_argument("--lote", type=int, default=20,
                    help="textos por chamada (padrão 20, igual ao de produção)")
    ap.add_argument("--pausa", type=float, default=0.5,
                    help="piso da pausa entre lotes, em segundos (padrão 0,5 = plano pago)")
    ap.add_argument("--tentativas", type=int, default=6,
                    help="repetições do mesmo lote diante de 429")
    ap.add_argument("--estimar", action="store_true",
                    help="só calcula quanto falta e sai, sem chamar a API")
    ap.add_argument("--limite", type=int,
                    help="embute no máximo N textos e para — para testar o "
                         "caminho da API sem pagar a conta inteira")
    args = ap.parse_args()

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    alvos = [args.so] if args.so else list(CONDICOES)

    # ── inventário: o que cada condição precisa, e o que é texto distinto ────
    registros: dict[str, list[dict]] = {}
    hashes: dict[str, list[str]] = {}
    por_hash: dict[str, str] = {}          # hash -> texto de embedding
    for nome in list(alvos):
        regs = carregar_condicao(nome)
        if not regs:
            alvos.remove(nome)
            continue
        registros[nome] = regs
        hs = []
        for r in regs:
            h = embedding_text_hash(r)
            hs.append(h)
            por_hash.setdefault(h, build_embedding_text(r))
        hashes[nome] = hs
        print(f"  {nome:6s} {len(regs):6d} trechos")

    total_trechos = sum(len(v) for v in registros.values())
    cache = carregar_cache()
    faltam = [h for h in por_hash if h not in cache]
    chars = sum(len(por_hash[h]) for h in faltam)

    print(f"\n  trechos ao todo        : {total_trechos}")
    print(f"  textos distintos       : {len(por_hash)}  "
          f"({1 - len(por_hash)/total_trechos:.0%} de economia pelo reaproveitamento)")
    print(f"  já no cache            : {len(por_hash) - len(faltam)}")
    print(f"  a calcular             : {len(faltam)}")
    print(f"  tokens a calcular      : {chars/CHARS_POR_TOKEN/1e6:.2f} M "
          f"(1 token ≈ {CHARS_POR_TOKEN} chars, medido neste corpus)")
    if faltam:
        lotes = -(-len(faltam) // args.lote)
        print(f"  lotes                  : {lotes} de {args.lote} "
              f"→ ~{lotes * args.pausa / 60:.1f} min de pausa no piso de {args.pausa}s")

    if args.estimar:
        return

    if args.limite:
        faltam = faltam[:args.limite]
        print(f"\n  --limite: só {len(faltam)} texto(s) nesta execução")

    # ── embutir o que falta ─────────────────────────────────────────────────
    if faltam:
        ritmo = Ritmo(piso=args.pausa)
        textos = [por_hash[h] for h in faltam]
        t0 = time.perf_counter()
        feitos = 0
        # Os bytes primeiro, o hash depois. Se cair no meio, sobra vetor sem
        # hash — que `carregar_cache` descarta. Na ordem inversa sobraria hash
        # sem vetor, e aí o cache devolveria lixo.
        with CACHE_VETORES.open("ab") as fv, CACHE_HASHES.open("a", encoding="utf-8") as fh:
            for inicio, vetores in embutir(textos, args.lote, ritmo, args.tentativas):
                for j, v in enumerate(vetores):
                    h = faltam[inicio + j]
                    v = normalizar(v)
                    cache[h] = v
                    fv.write(v.tobytes())
                    fv.flush()
                    fh.write(h + "\n")
                feitos += len(vetores)
                fh.flush()
                if feitos % (args.lote * 10) == 0 or feitos == len(faltam):
                    dt = time.perf_counter() - t0
                    resta = (len(faltam) - feitos) * dt / max(feitos, 1)
                    print(f"  {feitos}/{len(faltam)}  {dt/60:.1f} min  "
                          f"falta ~{resta/60:.1f} min  pausa {ritmo.atual:.2f}s", flush=True)
        print(f"\n  {feitos} vetores em {(time.perf_counter()-t0)/60:.1f} min "
              f"| {ritmo.total_429} vez(es) no limite")

    # ── montar um .npz por condição ─────────────────────────────────────────
    print()
    for nome in alvos:
        hs = hashes[nome]
        ausentes = [h for h in hs if h not in cache]
        if ausentes:
            print(f"  {nome}: faltam {len(ausentes)} vetores — rode de novo")
            continue

        matriz = np.stack([cache[h] for h in hs]).astype("float32")
        normas = np.linalg.norm(matriz, axis=1)
        assert matriz.shape == (len(hs), DIM), f"forma inesperada: {matriz.shape}"
        assert np.allclose(normas, 1.0, atol=1e-3), "vetor não normalizado"
        assert not np.isnan(matriz).any(), "vetor com NaN"

        destino = RESULTADOS / f"emb_gemini_{nome}.npz"
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


if __name__ == "__main__":
    main()
