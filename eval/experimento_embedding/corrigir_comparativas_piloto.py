#!/usr/bin/env python3
"""Repara as comparativas do piloto cujo gabarito não sustenta a resposta.

O DEFEITO. Revisando as 25 comparativas do piloto em 29/09/2026, 13 tinham
defeito e 5 eram limítrofes. Dois tipos:

  ESTRUTURAL   a pergunta nomeia duas normas e o gabarito cita uma só. A
               resposta afirma coisas sobre a norma ausente sem nenhuma
               citação. Quatro casos: q0081, q0085, q0093, q0099.
  DE CONTEÚDO  a citação existe mas é outra frase — do artigo certo, às vezes
               adjacente à certa — e não sustenta o que a resposta diz. O caso
               extremo é a FDA em q0092 e q0095, onde a citação é moldura pura
               ("a seção seguinte descreve as informações que os patrocinadores
               devem fornecer") e a resposta fala de sistema de qualidade.

Isso não era falha de busca: o recuperador estava sendo cobrado por não achar
texto que não responde à pergunta. As comparativas do piloto marcavam 2 de 25
contra 12 de 50 das geradas, e parte dessa diferença é gabarito, não busca.

POR QUE UM SCRIPT SEPARADO, E NÃO EDITAR O JSON. `perguntas_piloto.json` é
gerado por `converter_piloto.py` a partir de `golden_qa.jsonl`. Editar o JSON à
mão faria a correção sumir na próxima conversão. Este script roda DEPOIS do
conversor e é idempotente — as âncoras são reextraídas do corpus toda vez, então
rodar duas vezes dá o mesmo resultado.

Uso:
  python eval/experimento_embedding/corrigir_comparativas_piloto.py
  python eval/experimento_embedding/corrigir_comparativas_piloto.py --aplicar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
DADOS = AQUI / "dados"
ALVO = DADOS / "perguntas_piloto.json"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

from lib_citacao import AncoraRuim, citar  # noqa: E402

JANELA = 420
ANTES = 90

# qid → {"ancoras": [(documento, âncora), ...], "resposta": str|None, "nota": str}
# `resposta` só quando o texto do corpus não sustenta a redação original e é mais
# honesto ajustar a resposta do que forçar uma âncora que não existe.
REPAROS: dict[str, dict] = {

    # ── estruturais: faltava o segundo documento ────────────────────────────
    "q0081": {
        "nota": "faltava o PL 2338; a citação do NIST era da função GOVERN e a resposta falava de MAP",
        "ancoras": [
            ("projeto_lei_2338_BR_2023",
             "O Poder Executivo é autorizado a estabelecer o Sistema Nacional de Regulação e Governança"),
            ("ai_risk_management_framework_NIST_2023",
             "After putting in place the structures, systems, processes, and teams described in the GOVERN function"),
        ],
        "resposta": "O PL 2338/2023 cria uma estrutura institucional: autoriza o Poder Executivo a estabelecer o Sistema Nacional de Regulação e Governança de Inteligência Artificial (SIA). O NIST AI RMF propõe uma estrutura interna às organizações, em que, depois de implantados os arranjos, sistemas, processos e equipes descritos na função GOVERN, a organização se beneficia de uma cultura orientada a propósito. Um governa por instituição; o outro, por processo interno.",
    },
    "q0085": {
        "nota": "faltava a Lei 14.874; a resposta afirmava atribuições do CEP sem citação",
        "ancoras": [
            ("codigo_etica_medica_CFM_2019",
             "Deixar de atuar com absoluta isenção quando designado para servir como perito ou como auditor"),
            ("lei_14874_BR_2024",
             "assegurar os direitos, a segurança e o bem-estar dos participantes da pesquisa, especialmente dos participantes em situação de vulnerabilidade"),
        ],
        "resposta": "O Código de Ética Médica veda ao médico deixar de atuar com absoluta isenção quando designado perito ou auditor, e ultrapassar os limites de suas atribuições e de sua competência. A Lei nº 14.874/2024 atribui ao Comitê de Ética em Pesquisa a responsabilidade de assegurar os direitos, a segurança e o bem-estar dos participantes da pesquisa, especialmente os em situação de vulnerabilidade. Ambos protegem quem está sujeito à avaliação de um terceiro, por deveres de isenção e de tutela.",
    },
    "q0093": {
        "nota": "faltava a OMS; a resposta afirmava provisões para vulneráveis sem citação",
        "ancoras": [
            ("projeto_lei_2338_BR_2023",
             "direito à não discriminação ilícita ou abusiva e à correção de vieses discriminatórios"),
            ("ethics_governance_ai_health_WHO_2021",
             "Special provision should be made to protect the rights and welfare of vulnerable persons"),
        ],
        "resposta": None,
    },
    "q0099": {
        "nota": "faltava a OMS; a citação nova trata de PPP, e a resposta foi ajustada ao que o texto diz",
        "ancoras": [
            ("european_health_data_space_EU_2025",
             "a investigação científica relacionada com os sectores da saúde ou da prestação de cuidados"),
            ("ethics_governance_ai_health_WHO_2021",
             "are usually designed by companies or through public"),
        ],
        "resposta": "O Espaço Europeu de Dados de Saúde arrola entre as finalidades de utilização secundária a investigação científica ligada aos setores da saúde e da prestação de cuidados, que contribua para a saúde pública ou para a avaliação das tecnologias da saúde. A OMS observa que essas tecnologias em saúde costumam ser concebidas por empresas ou por parcerias público-privadas, ainda que muitos governos também as desenvolvam e implantem, e que algumas das maiores empresas de tecnologia do mundo desenvolvem tais aplicações. Um cria o regime de acesso; o outro chama atenção para quem, na prática, constrói as soluções.",
    },

    # ── de conteúdo: a citação não sustentava a resposta ────────────────────
    "q0078": {
        "nota": "a citação da LGPD era sobre exigir consentimento para compartilhar; a resposta fala da dispensa",
        "ancoras": [
            ("resolucao_2314_CFM_2022",
             "deve ser assegurado consentimento explícito"),
            ("lgpd_BR_2018",
             "É dispensada a exigência do consentimento previsto no caput deste artigo para os dados tornados manifestamente públicos"),
        ],
        "resposta": None,
    },
    "q0082": {
        "nota": "a citação da ANPD só falava em reparação; a resposta afirmava inversão do ônus e regresso",
        "ancoras": [
            ("resolucao_19_ANPD_2024",
             "o juiz poderá inverter o ônus da prova a favor do Titular"),
            ("gdpr_regulation_EU_2016",
             "When deciding whether to impose an administrative fine and deciding on the amount"),
        ],
        "resposta": "A Resolução ANPD nº 19/2024 prevê que, nos termos da legislação nacional, o juiz poderá inverter o ônus da prova a favor do titular quando a alegação for verossímil ou houver hipossuficiência. O GDPR determina que, ao decidir sobre impor uma coima e sobre o seu montante em cada caso, se dê a devida atenção aos fatores previstos. Um desloca o ônus probatório para proteger o titular; o outro disciplina a dosimetria da sanção.",
    },
    "q0083": {
        "nota": "a citação da FDA era um marcador solto; a do GDPR era 'ao avaliar o nível de segurança'",
        "ancoras": [
            ("ai_device_software_guidance_FDA_2025",
             "validating, authenticating, and cleansing data"),
            ("gdpr_regulation_EU_2016",
             "the ability to ensure the ongoing confidentiality, integrity, availability and resilience"),
        ],
        "resposta": None,
    },
    "q0084": {
        "nota": "a citação da OMS era sobre implementação por stakeholders; a resposta fala de responsividade",
        "ancoras": [
            ("ethics_governance_ai_health_WHO_2021",
             "Responsiveness requires that designers, developers and users continuously, systematically and transparently assess"),
            ("ethical_impact_assessment_UNESCO_2023",
             "Accountability and Responsibility The Recommendation highlights"),
        ],
        "resposta": None,
    },
    "q0092": {
        "nota": "a citação da FDA era moldura pura; a resposta fala de sistema de qualidade",
        "ancoras": [
            ("ai_device_software_guidance_FDA_2025",
             "nonconforming product (21 CFR 820.90), and corrective and preventive action"),
            ("resolucao_2314_CFM_2022",
             "considera o atendimento presencial como padrão ouro de referência"),
        ],
        "resposta": None,
    },
    "q0095": {
        "nota": "mesma citação-moldura da FDA de q0092",
        "ancoras": [
            ("projeto_lei_2338_BR_2023",
             "Os agentes de IA de alto risco devem garantir que seus sistemas estão de acordo com as medidas de governança"),
            ("ai_device_software_guidance_FDA_2025",
             "nonconforming product (21 CFR 820.90), and corrective and preventive action"),
        ],
        "resposta": None,
    },
    "q0097": {
        "nota": "a citação do GDPR era 'ao avaliar o nível'; a lista de medidas está na frase anterior",
        "ancoras": [
            ("gdpr_regulation_EU_2016",
             "the ability to ensure the ongoing confidentiality, integrity, availability and resilience"),
            ("resolucao_19_ANPD_2024",
             "devem ser adotadas integralmente e sem qualquer alteração em seu texto"),
        ],
        "resposta": None,
    },
    "q0098": {
        "nota": "a citação do NIST era sobre consequências graves; a resposta fala de notificação",
        "ancoras": [
            ("projeto_lei_2338_BR_2023",
             "direito à informação quanto às suas interações com sistemas de IA"),
            ("ai_risk_management_framework_NIST_2023",
             "how a human operator or user is notified when a potential or actual adverse outcome"),
        ],
        "resposta": None,
    },

    # ── limítrofes: âncora adjacente à que sustenta ────────────────────────
    "q0076": {
        "nota": "a resposta afirma que transparência não garante precisão; a frase estava fora da janela",
        "ancoras": [
            ("ai_risk_management_framework_NIST_2023",
             "A transparent system is not necessarily an accurate"),
            ("ethical_impact_assessment_UNESCO_2023",
             "Accountability and Responsibility The Recommendation highlights"),
        ],
        "resposta": None,
    },
    "q0094": {
        "nota": "faltava a parte das escolhas do paciente; uma só passagem do CEM cobre as duas",
        "ancoras": [
            ("ethical_impact_assessment_UNESCO_2023",
             "decisions are understood to have an impact that is irreversible"),
            ("codigo_etica_medica_CFM_2019",
             "o médico aceitará as escolhas de seus pacientes relativas aos procedimentos"),
        ],
        "resposta": None,
    },
    "q0096": {
        "nota": "a resposta fala de falta de métricas consensuais; a citação era sobre contexto",
        "ancoras": [
            ("ai_risk_management_framework_NIST_2023",
             "the current lack of consensus on robust and verifiable measurement methods"),
            ("ethical_impact_assessment_UNESCO_2023",
             "GRAVITY LEVEL: Critical"),
        ],
        "resposta": None,
    },

    # ── citação intrusa ────────────────────────────────────────────────────
    "q0086": {
        "nota": "tinha 4 citações; a quarta (situações terminais) é evidência de q0094",
        "ancoras": [
            ("resolucao_19_ANPD_2024",
             "o juiz poderá inverter o ônus da prova a favor do Titular"),
            ("resolucao_19_ANPD_2024",
             "A Parte que reparar o dano ao titular tem direito de regresso"),
            ("codigo_etica_medica_CFM_2019",
             "O médico se responsabilizará, em caráter pessoal e nunca presumido"),
        ],
        "resposta": None,
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    perguntas = json.loads(ALVO.read_text(encoding="utf-8"))
    por_n = {str(r["n"]): r for r in perguntas}

    falhas, feitos = [], []
    for qid, rep in REPAROS.items():
        r = por_n.get(qid)
        if r is None:
            falhas.append((qid, "pergunta não existe em perguntas_piloto.json"))
            continue
        ev, docs, erro = [], [], None
        for doc, ancora in rep["ancoras"]:
            try:
                ev.append(citar(doc, ancora, janela=JANELA, antes=ANTES))
                docs.append(doc)
            except AncoraRuim as e:
                erro = str(e)
                break
        if erro:
            falhas.append((qid, erro))
            continue
        feitos.append((qid, r, ev, docs, rep))

    print(f"reparos: {len(feitos)} de {len(REPAROS)}")
    if falhas:
        print(f"\n✗ âncoras a corrigir: {len(falhas)}")
        for qid, e in falhas:
            print(f"   [{qid}] {e}")

    print("\n" + "═" * 78)
    for qid, r, ev, docs, rep in feitos:
        print(f"\n[{qid}] {rep['nota']}")
        print(f"  PERGUNTA: {r['question']}")
        resp = rep["resposta"] or r["answer"]
        marca = " (REESCRITA)" if rep["resposta"] else ""
        print(f"  RESPOSTA{marca}: {resp[:300]}")
        for doc, cit in zip(docs, ev):
            print(f"  ── {doc}")
            print(f"     {cit[:300]}")

    if not args.aplicar:
        print("\n(relatório apenas — use --aplicar para gravar)")
        return
    if falhas:
        print("\n✗ não gravei: corrija as âncoras primeiro.")
        return

    for qid, r, ev, docs, rep in feitos:
        r["evidencia"] = ev
        r["n_evidencias"] = len(ev)
        unicos = sorted(set(docs))
        r["documento"] = unicos[0] if len(unicos) == 1 else None
        r["documentos"] = unicos if len(unicos) > 1 else None
        if rep["resposta"]:
            r["answer"] = rep["resposta"]
    ALVO.write_text(json.dumps(perguntas, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✓ {ALVO}")
    print("  Em seguida: montar_acervo.py --aplicar · gerar_qrels.py --aplicar")


if __name__ == "__main__":
    main()
