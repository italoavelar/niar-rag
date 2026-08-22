"""
eval/generation_eval/_common.py
═══════════════════════════════
Infra compartilhada pelos cenários de GERAÇÃO (um script por cenário nesta pasta).
Cada cenário, para cada variante de ICL {sem, com}:
    recupera o contexto (rankings da recuperação) → Qwen gera → 2 juízes pontuam
e grava (tag = <cenário>_icl{0,1}):
    results/generation/judged_<tag>.jsonl   bruto: resposta + notas + justificativas
    results/generation/scenario_<tag>.csv   organizado: resposta, contexto, médias

Cenários (a tag SEMPRE começa pelo nome do sistema de recuperação que a alimentou,
o que mantém o mapeamento config→sistema do 05_report correto):
    A_bm25[_mt]       geração a partir do BM25 (qual: retrieval.bm25.bm25_for_gen)
    B_dense_<emb>     geração a partir do denso (emb = retrieval.fusion.dense_for_fusion)
    C_fusion          geração a partir da fusão de deploy
    no_rag            SEM recuperação — Qwen sozinho (mede o valor da ferramenta)
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL))
from lib.common import (load_config, load_corpus, load_jsonl, save_jsonl, resolve,  # noqa: E402
                        ensure_dir, set_seed, setup_io)
from lib.rag_runner import RagRunner                                                # noqa: E402
from lib.judges import build_judges, judge_answer, run_judging                      # noqa: E402


def gen_out_dir(cfg):
    return ensure_dir(resolve(cfg["paths"]["results_dir"]) / "generation")


class Setup:
    """Carrega UMA vez (corpus, gold, rankings, gerador, juízes) e é compartilhado
    pelos cenários — evita recarregar tudo a cada um."""
    def __init__(self, cfg=None, limit=None, no_judge=False):
        self.cfg = cfg or load_config()
        setup_io(); set_seed(self.cfg.get("seed", 42))
        self.corpus = load_corpus(resolve(self.cfg["paths"]["corpus"]))
        qa = load_jsonl(resolve(self.cfg["paths"]["golden_qa"]))
        self.qa = qa[:limit] if limit else qa
        rpath = resolve(self.cfg["paths"]["results_dir"]) / "retrieval" / "rankings.json"
        if rpath.exists():
            self.rankings_all = json.loads(rpath.read_text(encoding="utf-8"))
        else:
            self.rankings_all = {}
            print("! aviso: rankings.json ausente — A/B/C exigem a recuperação; só o no_rag roda.")
        self.runner = RagRunner(self.cfg["generation"]["generator"], self.corpus,
                                self.cfg["generation"]["context_top_k"])
        self.criteria = self.cfg["judges"]["criteria"]
        self.scale = self.cfg["judges"]["scale"]
        self.judges = None if no_judge else build_judges(self.cfg)
        self.icl_variants = [bool(x) for x in self.cfg["generation"]["icl"]["variants"]]
        self.n_shots = self.cfg["generation"]["icl"]["n_shots"]
        self.dense_for_gen = self.cfg["retrieval"]["fusion"]["dense_for_fusion"]
        self.bm25_for_gen = (self.cfg["retrieval"].get("bm25", {})
                             .get("bm25_for_gen", "A_bm25"))

    def rankings(self, system):
        return self.rankings_all.get(system)


def _scenario_csv(S, tag, answers, od):
    """CSV organizado do cenário: 1 linha por pergunta com a resposta, o contexto
    recuperado, a referência e a nota de CADA juiz por critério (colunas
    <juiz>·<critério>), mais a MÉDIA dos juízes por critério."""
    crit = S.criteria
    judges = [m["name"] for m in S.cfg["judges"]["models"]]
    score_cols = ([f"{j}·{c}" for j in judges for c in crit]
                  + [f"média·{c}" for c in crit])
    with (od / f"scenario_{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "tipo", "deveria_responder", "question", "resposta_gerada",
                    "contexto_chunks", "resposta_referencia"] + score_cols)
        for a in answers:
            sc = a.get("scores", {}) or {}
            per_judge = []
            for j in judges:
                js = sc.get(j) if isinstance(sc.get(j), dict) else {}
                for c in crit:
                    v = js.get(c)
                    per_judge.append(v if isinstance(v, (int, float)) else "")
            means = []
            for c in crit:
                vals = [sc[j].get(c) for j in judges
                        if isinstance(sc.get(j), dict) and isinstance(sc[j].get(c), (int, float))]
                means.append(round(sum(vals) / len(vals), 2) if vals else "")
            w.writerow([a.get("qid"), a.get("question_type"), a.get("should_answer"),
                        a.get("question"), a.get("answer"),
                        ", ".join(a.get("context_chunk_ids", []) or []),
                        a.get("reference_answer")] + per_judge + means)


def _load_partial(path):
    """Lê o checkpoint .partial.jsonl -> {qid: registro}. Ignora a última linha se
    ficou truncada por interrupção no meio da escrita."""
    done = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[r["qid"]] = r
    return done


def run_and_save(S, tag, rankings, icl, no_rag=False):
    """Gera + julga PERGUNTA A PERGUNTA, gravando cada registro concluído em
    judged_<tag>.partial.jsonl antes de seguir. Se a cota de API estourar (ou o
    processo cair), basta re-rodar o MESMO comando: as perguntas já concluídas são
    puladas e a execução retoma de onde parou. O .partial é apagado ao final.

    Um registro só entra no checkpoint se a geração E os dois juízes tiverem
    sucedido — respostas com juiz em erro voltam a ser tentadas na próxima rodada.
    """
    from tqdm import tqdm
    od = gen_out_dir(S.cfg)
    final = od / f"judged_{tag}.jsonl"
    alvo = {r["qid"] for r in S.qa}
    if final.exists():
        feitas = load_jsonl(final)
        if alvo <= {r["qid"] for r in feitas}:
            print(f"══ {tag}: já concluído ({len(feitas)} respostas) — pulando. "
                  f"Apague {final.name} para refazer.")
            return feitas

    part = od / f"judged_{tag}.partial.jsonl"
    done = _load_partial(part)

    print(f"══ Geração {tag}  ({'SEM contexto' if no_rag else 'com RAG'}) ══")
    if done:
        print(f"  ↻ retomando: {len(done)}/{len(S.qa)} perguntas já concluídas")

    answers = []
    with part.open("a", encoding="utf-8") as fh:
        for r in tqdm(S.qa, desc=f"[{tag}]", leave=False):
            if r["qid"] in done:
                answers.append(done[r["qid"]])
                continue
            if no_rag:
                res = S.runner.answer_no_rag(r["question"], icl=icl, n_shots=S.n_shots)
            else:
                res = S.runner.answer(r["question"], (rankings or {}).get(r["qid"], []),
                                      icl=icl, n_shots=S.n_shots)
            rec = {**res, "qid": r["qid"], "question": r["question"],
                   "reference_answer": r.get("reference_answer", ""),
                   "question_type": r.get("question_type", ""),
                   "should_answer": bool(r.get("qrels"))}
            ok = True
            if S.judges:
                # julga ANTES de descartar context_text (o juiz precisa do contexto)
                rec["scores"] = {name: judge_answer(cli, rec, S.criteria, S.scale)
                                 for name, cli in S.judges}
                ok = not any("_error" in s for s in rec["scores"].values())
            rec.pop("context_text", None)      # remove o texto longo antes de salvar
            answers.append(rec)
            if ok:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            # reescreve o CSV a cada pergunta p/ acompanhar o progresso durante o run.
            # Custo irrisório perto de uma chamada de API; e se o arquivo estiver
            # aberto no Excel a escrita falha — ignoramos, para não derrubar a rodada.
            try:
                _scenario_csv(S, tag, answers, od)
            except OSError:
                pass

    save_jsonl(answers, od / f"judged_{tag}.jsonl")
    _scenario_csv(S, tag, answers, od)
    # Só descarta o checkpoint se TUDO deu certo. Sem isso, uma rodada que chega
    # às 100 perguntas com juízes em erro apagava o .partial e "concluía" com os
    # defeitos embutidos no arquivo final.
    ruins = sum(1 for a in answers
                if any("_error" in s for s in (a.get("scores") or {}).values()))
    if ruins:
        print(f"  ! {ruins} resposta(s) com juiz em erro — checkpoint preservado.")
        print(f"    Conserte com: python tools/repair_generation.py --tag {tag}")
    else:
        part.unlink(missing_ok=True)
    print(f"  ✓ {len(answers)} respostas → judged_{tag}.jsonl + scenario_{tag}.csv")
    return answers
