#!/usr/bin/env python3
"""Confere as perguntas geradas contra a passagem que as gerou E contra o corpus.

A TRAVA. Cada citação de `evidencia` tem de aparecer LITERALMENTE em dois
lugares, e as duas checagens perguntam coisas diferentes:

    na passagem   a pergunta saiu MESMO daquele trecho, não de memória.
                  Pega alucinação na hora da geração.

    no corpus     a evidência é RECUPERÁVEL. Pega o caso silencioso e pior:
                  citação fiel a uma passagem defasada, que já não existe no
                  corpus que vai ser embutido. O gabarito passaria a cobrar
                  texto inexistente e toda condição erraria igual.

A segunda faltava, e o estrago apareceu: 3 perguntas boas foram rejeitadas por
"citação ausente da passagem" quando a citação estava no corpus — só que 1.378
caracteres antes de onde o índice `inicio` dizia. Ver `reancorar_passagens.py`.

A COMPARAÇÃO É CONTRA O DOCUMENTO RECONSTRUÍDO, nunca contra um dos recortes sob
teste. Exigir que a citação caiba dentro de UM trecho favoreceria o recorte
grande — e se a evidência é partida pelo corte é justamente isso que o
experimento tem de medir, não de esconder.

Normaliza espaço, acento e caixa: a passagem vem de PDF e traz hifenização e
quebra de linha em posições arbitrárias ("authoris­ ation"). Exige-se que o
CONTEÚDO seja o mesmo, não os bytes.

Confere também:
  - pergunta não vazia e sem expressão meta ("segundo o texto")
  - resposta não vazia
  - `n_evidencias` bate com o número de citações
  - nenhuma pergunta repetida

Uso:
  python eval/experimento_embedding/validar_perguntas.py
  python eval/experimento_embedding/validar_perguntas.py --consolidar
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
DADOS = AQUI / "dados"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

META = re.compile(
    r"(segundo o (texto|trecho|documento|parágrafo)|de acordo com o (texto|trecho)|"
    r"n[oa] (texto|trecho|documento) (acima|citado|apresentado)|conforme o (texto|trecho))",
    re.IGNORECASE)


def normalizar(t: str) -> str:
    """Espaço colapsado, acento e caixa removidos, hifenização de PDF desfeita."""
    t = (t or "").replace("\u00ad", "")
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--consolidar", action="store_true",
                    help="junta os lotes válidos em perguntas.json")
    args = ap.parse_args()

    import bench_tamanho as B

    passagens = {p["n"]: p for p in json.loads(
        (DADOS / "passagens.json").read_text(encoding="utf-8"))["passagens"]}
    por_doc = B.carregar_corpus()
    documentos: dict[str, str] = {}

    def texto_do_documento(doc_id: str) -> str:
        if doc_id not in documentos:
            documentos[doc_id] = normalizar(B.reconstruir(por_doc[doc_id]))
        return documentos[doc_id]

    lotes = sorted(DADOS.glob("perguntas_lote*.json"))
    if not lotes:
        sys.exit(f"nenhum lote em {DADOS}")

    aceitos, recusas, rejeitados, vistas = [], [], [], set()
    for lote in lotes:
        for r in json.loads(lote.read_text(encoding="utf-8")):
            n = r.get("n")
            p = passagens.get(n)
            if not r.get("question"):
                recusas.append((n, r.get("motivo", "")))
                continue

            q = r["question"].strip()
            if (m := META.search(q)):
                rejeitados.append((n, f"expressão meta: {m.group(0)!r}"))
                continue
            if normalizar(q) in vistas:
                rejeitados.append((n, "pergunta duplicada"))
                continue
            if not (r.get("answer") or "").strip():
                rejeitados.append((n, "sem resposta"))
                continue

            ev = [e for e in (r.get("evidencia") or []) if e.strip()]
            if not ev:
                rejeitados.append((n, "sem evidência"))
                continue

            # Provenance: a passagem congelada na própria pergunta é a fonte de
            # verdade; `passagens.json` é só o sorteio de onde ela saiu.
            fonte = r.get("passagem_texto") or (p or {}).get("texto")
            if fonte is None:
                rejeitados.append((n, "sem passagem — rode reancorar_passagens.py"))
                continue
            fonte = normalizar(fonte)
            if (faltando := [e for e in ev if normalizar(e) not in fonte]):
                rejeitados.append((n, f"citação ausente da passagem: {faltando[0][:58]!r}"))
                continue

            doc_id = r.get("documento") or (p or {}).get("documento")
            if doc_id not in por_doc:
                rejeitados.append((n, f"documento fora do corpus: {doc_id}"))
                continue
            alvo = texto_do_documento(doc_id)
            if (faltando := [e for e in ev if normalizar(e) not in alvo]):
                rejeitados.append((n, f"citação ausente do CORPUS: {faltando[0][:58]!r}"))
                continue

            if r.get("n_evidencias") not in (None, len(ev)):
                rejeitados.append((n, f"n_evidencias={r['n_evidencias']} mas há {len(ev)} citações"))
                continue

            vistas.add(normalizar(q))
            aceitos.append({**r, "documento": doc_id,
                            "forma": r.get("forma") or (p or {}).get("forma"),
                            "janela": r.get("janela") or (p or {}).get("janela"),
                            "n_evidencias": len(ev)})

    print(f"lotes lidos: {len(lotes)}")
    print(f"  aceitos               : {len(aceitos)}")
    print(f"  recusados na geração  : {len(recusas)}")
    for n, motivo in recusas:
        print(f"      n={n}: {motivo[:78]}")
    print(f"  REJEITADOS na validação: {len(rejeitados)}")
    for n, motivo in rejeitados:
        print(f"      n={n}: {motivo}")

    if aceitos:
        import collections
        print(f"\n  por janela     : {dict(sorted(collections.Counter(a['janela'] for a in aceitos).items()))}")
        print(f"  por forma      : {dict(collections.Counter(a['forma'] for a in aceitos))}")
        print(f"  por documento  : {len({a['documento'] for a in aceitos})} distintos")
        print(f"  por nº de evidências: {dict(sorted(collections.Counter(a['n_evidencias'] for a in aceitos).items()))}")

    if args.consolidar and aceitos:
        destino = DADOS / "perguntas.json"
        destino.write_text(json.dumps(aceitos, ensure_ascii=False, indent=1),
                           encoding="utf-8")
        print(f"\n✓ {destino}  ({len(aceitos)} perguntas)")


if __name__ == "__main__":
    main()
