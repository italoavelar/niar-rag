#!/usr/bin/env python3
"""Factuais adicionais: uma pergunta, um fato, uma citação.

POR QUE ESTE LOTE EXISTE, e por que nestes documentos. O acervo tinha 39
factuais concentradas em poucos documentos — 7 só do UNESCO, 5 do PL 2338 — e
zero nos maiores: a Constituição com 1 milhão de caracteres, o ECA, o Código
Penal, o Plano Brasileiro de IA, os documentos da OMS. Uma factual é a pergunta
mais simples do acervo e por isso a mais reveladora de falha de recuperação
básica; não tê-la nos documentos grandes deixava um buraco onde ela mais
importa.

A alocação segue o tamanho do documento e a ausência de cobertura, não o que
seria mais fácil de escrever.

UMA CITAÇÃO SÓ, e ela tem de bastar. Se a resposta precisa de duas, a pergunta é
multi-hop e está no arquivo errado — a distinção é o que torna a métrica de
completude interpretável por tipo.

As citações são extraídas do corpus por `lib_citacao.citar`, a partir de âncora
curta. Não se transcreve nada.

Uso:
  python eval/experimento_embedding/gerar_factuais.py
  python eval/experimento_embedding/gerar_factuais.py --aplicar
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
DADOS = AQUI / "dados"
SAIDA = DADOS / "perguntas_factuais.json"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

from lib_citacao import AncoraRuim, citar  # noqa: E402

# (documento, âncora, pergunta, resposta, dificuldade)
ESPEC: list[tuple[str, str, str, str, str]] = [

    # ── Constituição ────────────────────────────────────────────────────────
    ("cf_BR_1988", "é inviolável o sigilo da correspondência",
     "Em que hipótese a Constituição admite quebra do sigilo das comunicações telefônicas?",
     "O sigilo da correspondência e das comunicações telegráficas, de dados e telefônicas é inviolável, salvo, no último caso, por ordem judicial, nas hipóteses e na forma que a lei estabelecer para fins de investigação criminal ou instrução processual penal.",
     "easy"),
    ("cf_BR_1988", "A assistência à saúde é livre à iniciativa privada",
     "A iniciativa privada pode atuar na assistência à saúde e em que condições participa do SUS?",
     "A assistência à saúde é livre à iniciativa privada. As instituições privadas podem participar de forma complementar do Sistema Único de Saúde, segundo as diretrizes deste, mediante contrato de direito público ou convênio, com preferência para entidades filantrópicas e sem fins lucrativos.",
     "medium"),
    ("cf_BR_1988", "São de relevância pública as ações e serviços de saúde",
     "Que natureza a Constituição atribui às ações e serviços de saúde?",
     "São de relevância pública as ações e serviços de saúde, cabendo ao poder público dispor, nos termos da lei, sobre sua regulamentação, fiscalização e controle.",
     "easy"),
    ("cf_BR_1988", "por invalidez permanente, sendo os proventos proporcionais",
     "Como se calculam os proventos do servidor aposentado por invalidez permanente?",
     "Os proventos são proporcionais ao tempo de contribuição, exceto se a invalidez decorrer de acidente em serviço, moléstia profissional ou doença grave, contagiosa ou incurável especificada em lei.",
     "medium"),

    # ── ECA e Código Penal ──────────────────────────────────────────────────
    ("eca_BR_1990", "É assegurado atendimento integral à saúde da criança",
     "Que atendimento de saúde o ECA assegura à criança e ao adolescente?",
     "É assegurado atendimento integral à saúde da criança e do adolescente por intermédio do Sistema Único de Saúde, garantido o acesso universal e igualitário às ações e serviços para promoção, proteção e recuperação da saúde.",
     "easy"),
    ("eca_BR_1990", "buscar refúgio, auxílio e orientação",
     "Que direitos de liberdade o ECA reconhece à criança e ao adolescente?",
     "Entre os aspectos do direito à liberdade, o ECA inclui buscar refúgio, auxílio e orientação.",
     "medium"),
    ("cp_BR_1940", "Violação do segredo profissional",
     "Qual a pena para quem revela, sem justa causa, segredo obtido em razão da profissão?",
     "Revelar, sem justa causa, segredo de que se tem ciência em razão de função, ministério, ofício ou profissão, quando a revelação possa produzir dano a outrem, é punido com detenção de três meses a um ano, ou multa.",
     "easy"),
    ("cp_BR_1940", "resulta lesão corporal de natureza grave ou gravíssima",
     "Qual a pena quando do induzimento ou instigação ao suicídio resulta lesão corporal grave?",
     "Quando da instigação, induzimento ou auxílio ao suicídio, ou da tentativa, resulta lesão corporal de natureza grave ou gravíssima, a pena é de reclusão de 1 a 3 anos.",
     "medium"),

    # ── Consumidor e SUS ────────────────────────────────────────────────────
    ("cdc_BR_1990", "no prazo de cinco dias úteis, comunicar a alteração",
     "Em quanto tempo o arquivista deve comunicar a correção de dado incorreto do consumidor?",
     "Feita a correção, o arquivista tem cinco dias úteis para comunicar a alteração aos eventuais destinatários das informações incorretas.",
     "easy"),
    ("cdc_BR_1990", "são considerados entidades de caráter público",
     "Que natureza jurídica o CDC atribui aos bancos de dados de consumidores?",
     "Os bancos de dados e cadastros relativos a consumidores, os serviços de proteção ao crédito e congêneres são considerados entidades de caráter público.",
     "easy"),
    ("lei_8080_BR_90", "obedecendo ainda aos seguintes princípios",
     "Qual o primeiro princípio que a lei do SUS enumera para os serviços de saúde?",
     "Além das diretrizes constitucionais, os serviços obedecem aos princípios enumerados na lei, começando pela universalidade de acesso aos serviços de saúde em todos os níveis de assistência.",
     "medium"),
    ("lei_8080_BR_90", "Estão incluídas ainda no campo de atuação do Sistema Único de Saúde",
     "Que atividades a lei inclui no campo de atuação do SUS além da assistência?",
     "O campo de atuação do SUS abrange, além da assistência, a execução de ações de vigilância sanitária, vigilância epidemiológica, saúde do trabalhador e assistência terapêutica integral, inclusive farmacêutica.",
     "medium"),

    # ── Desidentificação e governança de dados ──────────────────────────────
    ("de_identifying_government_datasets_NIST_2023", "Ongoing monitoring is in place",
     "O que o NIST espera que exista depois que uma estratégia de desidentificação é adotada?",
     "Espera-se monitoramento contínuo, para assegurar a eficácia permanente da estratégia de desidentificação.",
     "medium"),
    ("de_identifying_government_datasets_NIST_2023", "privacy loss budget",
     "O que é um orçamento de perda de privacidade?",
     "É um limite superior para a perda de privacidade cumulativa total dos indivíduos.",
     "easy"),

    # ── Estratégia e plano nacionais ────────────────────────────────────────
    ("plano_brasileiro_ia_BR_2025", "investimentos de R$ 23 bilhões até 2028",
     "Quanto o Plano Brasileiro de Inteligência Artificial prevê investir e até quando?",
     "O PBIA prevê investimentos de R$ 23 bilhões até 2028, provenientes de fontes como crédito, recursos públicos e contrapartida de investimento privado, direcionados a infraestrutura tecnológica, capacitação de profissionais e fomento à pesquisa e inovação.",
     "easy"),
    ("plano_brasileiro_ia_BR_2025", "inclusão social e oferecendo soluções tangíveis",
     "Qual o objetivo declarado do Plano Brasileiro de Inteligência Artificial?",
     "Garantir que a inteligência artificial melhore a vida do povo brasileiro, promovendo inclusão social e oferecendo soluções tangíveis em áreas prioritárias como saúde e educação.",
     "easy"),
    ("plano_brasileiro_ia_BR_2025", "soberania digital",
     "Que conceito o Plano Brasileiro de IA associa ao seu propósito estratégico?",
     "O plano se apresenta como instrumento de soberania digital, e não apenas como plano tecnológico.",
     "medium"),

    # ── OMS ─────────────────────────────────────────────────────────────────
    ("regulatory_considerations_ai_health_WHO_2024", "18 regulatory considerations",
     "Quantas considerações regulatórias a OMS recomenda que as partes interessadas levem em conta na IA em saúde?",
     "A publicação recomenda que as partes interessadas considerem 18 considerações regulatórias ao desenvolver quadros e boas práticas para o uso de IA na assistência à saúde.",
     "easy"),
    ("regulatory_considerations_ai_health_WHO_2024", "documentation and transparency, risk management",
     "Que áreas temáticas gerais a OMS cobre nas considerações regulatórias sobre IA em saúde?",
     "As áreas são documentação e transparência, gestão de risco e abordagem de ciclo de vida de desenvolvimento, uso pretendido e validação analítica e clínica, qualidade dos dados relacionados à IA, privacidade e proteção, e engajamento e colaboração.",
     "medium"),
    ("regulatory_considerations_ai_health_WHO_2024", "listing of key regulatory considerations",
     "Que natureza a própria OMS atribui ao documento de considerações regulatórias?",
     "O documento se apresenta como uma listagem de considerações regulatórias chave e como recurso a ser considerado, não como norma vinculante.",
     "medium"),
    ("global_digital_health_strategy_WHO_2020_2027", "eHealth standardization and interoperability",
     "Que tema as resoluções regionais citadas pela estratégia global de saúde digital pediram aos Estados-Membros que desenvolvessem?",
     "As resoluções sobre padronização e interoperabilidade em saúde eletrônica urgiram os Estados-Membros a considerar o desenvolvimento de políticas e mecanismos legislativos ligados a uma estratégia nacional global de saúde eletrônica.",
     "medium"),
    ("global_digital_health_strategy_WHO_2020_2027", "smart wearables, platforms, tools enabling data exchange",
     "Que tecnologias a estratégia global de saúde digital lista como comprovadamente úteis ao contínuo de cuidado?",
     "Entre elas, monitoramento remoto, inteligência artificial, análise de big data, blockchain, dispositivos vestíveis inteligentes, plataformas e ferramentas que permitem troca e armazenamento de dados e captura remota de dados, criando um contínuo de cuidado.",
     "medium"),
    ("multimodal_models_guidance_WHO_2024", "designed to perform well-defined tasks",
     "Que exigência a OMS faz aos desenvolvedores quanto ao escopo das tarefas de um modelo multimodal?",
     "Os desenvolvedores devem garantir que os modelos sejam desenhados para executar tarefas bem definidas, com a acurácia e a confiabilidade necessárias para melhorar a capacidade dos sistemas de saúde e promover os interesses dos pacientes, sendo capazes de prever e entender resultados secundários potenciais.",
     "medium"),
    ("multimodal_models_guidance_WHO_2024", "premortems",
     "Que técnicas a OMS menciona para antecipar falhas no desenvolvimento de modelos multimodais?",
     "São mencionados os premortems e o red teaming como técnicas para atender à exigência de prever e compreender resultados secundários potenciais.",
     "medium"),
    ("ethics_governance_ai_health_WHO_2021", "The public should be engaged in the development of AI for health",
     "Que papel a OMS atribui ao público no desenvolvimento da IA para saúde?",
     "O público deve ser envolvido no desenvolvimento da IA para saúde, para compreender as formas de compartilhamento e uso de dados, opinar sobre que formas de IA são social e culturalmente aceitáveis e expressar suas preocupações e expectativas; a literacia em IA da população deve ser melhorada para que possa determinar quais tecnologias são aceitáveis.",
     "medium"),
    ("ethics_governance_ai_health_WHO_2021", "neither understand how an AI technology arrives",
     "Por que a transferência de decisões a máquinas ameaça a autonomia humana, segundo a OMS?",
     "Porque os humanos podem não compreender como a tecnologia de IA chega a uma decisão, nem conseguir negociar com ela para alcançar uma decisão compartilhada.",
     "medium"),

    # ── Conselhos de Medicina ───────────────────────────────────────────────
    ("resolucao_2306_CFM_2022", "termo de ajustamento de conduta (TAC), quando pertinente",
     "Que desfechos pode ter uma sindicância no processo ético-profissional?",
     "A sindicância pode resultar em termo de ajustamento de conduta quando pertinente, arquivamento quando não houver indícios de materialidade e autoria de infração ao Código de Ética Médica, ou instauração de processo ético-profissional quando esses indícios existirem.",
     "medium"),
    ("resolucao_2306_CFM_2022", "no prazo de 15 (quinze) dias corridos",
     "Quanto tempo o denunciante tem para recorrer do arquivamento de uma sindicância?",
     "Quinze dias corridos, contados da juntada aos autos do comprovante de ciência da intimação, em recurso dirigido ao presidente do CRM, que o remeterá ao CFM. O médico é intimado para, querendo, apresentar contrarrazões no mesmo prazo.",
     "medium"),
    ("resolucao_1627_CFM_2001", "todo procedimento técnico- profissional praticado por médico legalmente habilitado",
     "Como o CFM define o ato profissional de médico?",
     "É todo procedimento técnico-profissional praticado por médico legalmente habilitado, dirigido à promoção da saúde e prevenção de enfermidades, à prevenção da evolução das enfermidades e às demais finalidades enumeradas na resolução.",
     "easy"),
    ("resolucao_1627_CFM_2001", "A responsabilidade é a faculdade pela qual uma pessoa ou organização responde",
     "Como a resolução do CFM define responsabilidade profissional?",
     "É a faculdade pela qual uma pessoa ou organização responde pelas suas decisões, sem referência à vontade de outrem, noção que implica as de capacidade e de liberdade.",
     "medium"),
    ("resolucao_2454_CFM_2026", "Esta resolução estabelece normas para a pesquisa",
     "Que atividades a resolução do CFM sobre inteligência artificial na medicina disciplina?",
     "Estabelece normas para a pesquisa, o desenvolvimento, a governança, a auditoria, o monitoramento, a capacitação e o uso responsável de soluções que adotem modelos, sistemas e aplicações de IA na área da medicina.",
     "easy"),
    ("resolucao_2454_CFM_2026", "deverão priorizar o desenvolvimento cooperativo",
     "Que forma de desenvolvimento de IA as instituições médicas devem priorizar?",
     "Devem priorizar o desenvolvimento cooperativo de modelos, sistemas e aplicações de IA, promovendo a interoperabilidade e a disseminação de tecnologias, códigos, bases de dados e boas práticas com outros órgãos e entidades.",
     "medium"),
    ("resolucao_2336_CFM_2023", "atribuir capacid",
     "O que a resolução de publicidade médica proíbe quanto à atribuição de capacidade a métodos e equipamentos?",
     "É vedado atribuir capacidade privilegiada, mesmo que o médico seja o único a fazê-la; a resolução veda ainda oferecer serviços por meio de consórcio e similares.",
     "medium"),

    # ── Dispositivos médicos e ética em pesquisa ────────────────────────────
    ("imdrf_samd_risk_framework_2014", "SaMD is defined as software intended to be used",
     "Como o IMDRF define software como dispositivo médico?",
     "Software como dispositivo médico é o software destinado a ser usado para um ou mais propósitos médicos e que cumpre esses propósitos sem ser parte de um dispositivo médico de hardware.",
     "easy"),
    ("imdrf_samd_clinical_evaluation_2017", "SaMD categories are based on the levels of impact",
     "Em que se baseiam as categorias de software como dispositivo médico na avaliação clínica do IMDRF?",
     "Baseiam-se nos níveis de impacto sobre o paciente ou a saúde pública, conforme a informação precisa fornecida pelo software seja importante para tratar ou diagnosticar, conduzir o manejo clínico ou informar o manejo clínico.",
     "medium"),
    ("regulamento_interno_CEP_2024", "Uma vez aprovado o projeto",
     "Que posição o comitê de ética assume depois de aprovar um projeto de pesquisa?",
     "Uma vez aprovado o projeto, o CEP-UFMG passa a ser corresponsável no que se refere aos aspectos éticos da pesquisa.",
     "easy"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    temas: dict[str, str] = {}
    for linha in (AQUI.parents[1] / "data/processed/documents.jsonl").open(encoding="utf-8"):
        r = json.loads(linha)
        temas.setdefault(r["metadata"]["document_id"], r["metadata"].get("theme"))

    EN = {"de_identifying_government_datasets_NIST_2023",
          "regulatory_considerations_ai_health_WHO_2024",
          "global_digital_health_strategy_WHO_2020_2027",
          "multimodal_models_guidance_WHO_2024",
          "ethics_governance_ai_health_WHO_2021",
          "imdrf_samd_risk_framework_2014",
          "imdrf_samd_clinical_evaluation_2017"}

    registros, falhas = [], []
    for i, (doc, ancora, pergunta, resposta, dif) in enumerate(ESPEC, start=1):
        try:
            ev = citar(doc, ancora)
        except AncoraRuim as erro:
            falhas.append((i, str(erro)))
            continue
        registros.append({
            "n": f"f{len(registros)+1:03d}",
            "question": pergunta,
            "answer": resposta,
            "evidencia": [ev],
            "n_evidencias": 1,
            "documento": doc,
            "forma": None,
            "janela": None,
            "passagem_texto": None,
            "question_type": "factual",
            "difficulty": dif,
            "theme": temas.get(doc),
            "source_lang": "en" if doc in EN else "pt",
            "origem": "factuais_geradas",
        })

    print(f"factuais montadas: {len(registros)} de {len(ESPEC)}")
    if falhas:
        print(f"\nâncoras a corrigir: {len(falhas)}")
        for i, e in falhas:
            print(f"   [{i}] {e}")
    print(f"\npor documento: {dict(Counter(r['documento'][:28] for r in registros).most_common())}")
    print(f"dificuldade  : {dict(Counter(r['difficulty'] for r in registros))}")

    if args.aplicar and not falhas:
        SAIDA.write_text(json.dumps(registros, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        print(f"\n✓ {SAIDA}")
    elif args.aplicar:
        print("\n✗ não gravei: corrija as âncoras primeiro.")
    else:
        print("\n(relatório apenas — use --aplicar para gravar)")


if __name__ == "__main__":
    main()
