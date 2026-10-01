#!/usr/bin/env python3
"""Multi-hop com evidência REALMENTE dispersa, dentro de um documento só.

POR QUE ESTAS EXISTEM. Em 27/09/2026 descobriu-se que 42 das 101 perguntas
rotuladas `multi_hop` tinham as duas citações DENTRO DO MESMO trecho de 1.200
caracteres — recuperar uma recuperava a outra. A causa era construtiva: as 91 do
experimento de recorte nasceram de uma passagem de ~1.200 caracteres, o tamanho
do trecho de produção, então as duas citações saíam da mesma janela por
definição. O efeito é grande: o Gemini acerta Junta@5 em 0,968 delas e em 0,167
das de evidência distante.

Corrigido o tipo, o acervo ficou com 60 multi_hop contra o alvo de 75. Estas
fecham a conta, e são escritas com a dispersão como REQUISITO, não como acaso.
Foram 15 em 27/09; a 16ª entrou em 30/09, quando o alvo do artigo passou a ser
75/75/75 e o tipo estava em 74 — ver o comentário na última entrada de ESPEC.

A DIFERENÇA PARA UMA COMPARATIVA. A comparativa cruza dois documentos; esta cruza
duas regiões distantes DO MESMO documento. É o caso em que o recuperador acha o
documento certo — e ainda assim falha, porque traz cinco trechos da mesma seção.
Nenhum outro tipo isola isso.

O SCRIPT SE RECUSA A GRAVAR se alguma pergunta tiver dispersão menor que 2
trechos. Sem essa trava, a próxima leva repetiria o erro em silêncio — foi
exatamente assim que as 42 entraram.

Uso:
  python eval/experimento_embedding/gerar_multihop.py
  python eval/experimento_embedding/gerar_multihop.py --aplicar
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
PROJECT_ROOT = AQUI.parents[1]
DADOS = AQUI / "dados"
SAIDA = DADOS / "perguntas_multihop.json"
CORPUS = PROJECT_ROOT / "data/processed/documents.jsonl"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

from lib_citacao import AncoraRuim, citar  # noqa: E402

DISPERSAO_MINIMA = 2

# (documento, âncora_a, âncora_b, pergunta, resposta)
ESPEC: list[tuple[str, str, str, str, str]] = [

    # ── LGPD ────────────────────────────────────────────────────────────────
    ("lgpd_BR_2018",
     "portabilidade dos dados a outro fornecedor de serviço ou produto, mediante requisição expressa",
     "as medidas que foram ou que serão adotadas para reverter ou mitigar os efeitos do prejuízo",
     "Que direito de portabilidade a LGPD assegura ao titular e o que o controlador precisa informar ao comunicar um incidente de segurança?",
     "A LGPD assegura ao titular a portabilidade dos seus dados a outro fornecedor de serviço ou produto, mediante requisição expressa e de acordo com a regulamentação da autoridade nacional, observados os segredos comercial e industrial. Na comunicação de incidente de segurança, entre os elementos exigidos está a indicação das medidas que foram ou que serão adotadas para reverter ou mitigar os efeitos do prejuízo."),

    ("lgpd_BR_2018",
     "finalidade: realização do tratamento para propósitos legítimos, específicos, explícitos e informados ao titular",
     "será o órgão central de interpretação desta Lei",
     "Como a LGPD enuncia o princípio da finalidade e qual é o papel interpretativo atribuído à ANPD?",
     "O princípio da finalidade exige que o tratamento se realize para propósitos legítimos, específicos, explícitos e informados ao titular. Quanto à ANPD, a Lei determina que ela articule sua atuação com outros órgãos e entidades com competências sancionatórias e normativas sobre proteção de dados, e que seja o órgão central de interpretação da Lei e do estabelecimento de normas e diretrizes para a sua implementação."),

    ("lgpd_BR_2018",
     "Aplica-se aos membros do Conselho Diretor, após o exercício do cargo",
     "terão mandato de 2 (dois) anos, permitida 1 (uma) recondução",
     "Que restrição recai sobre os membros do Conselho Diretor da ANPD depois do cargo e qual é o mandato dos membros do Conselho Nacional de Proteção de Dados Pessoais e da Privacidade?",
     "Aos membros do Conselho Diretor da ANPD, após o exercício do cargo, aplica-se o disposto no art. 6º da Lei nº 12.813/2013, e a infração a esse dever caracteriza ato de improbidade administrativa. Já os membros do Conselho Nacional de Proteção de Dados Pessoais e da Privacidade têm mandato de dois anos, permitida uma recondução."),

    # ── Lei 8.080 ───────────────────────────────────────────────────────────
    ("lei_8080_BR_90",
     "a identificação e divulgação dos fatores condicionantes e determinantes da saúde",
     "organização de um sistema de formação de recursos humanos em todos os níveis de ensino",
     "Que objetivo do SUS a Lei nº 8.080 enuncia primeiro e que objetivo a política de recursos humanos da saúde deve cumprir?",
     "Entre os objetivos do SUS, a Lei nº 8.080 enuncia a identificação e divulgação dos fatores condicionantes e determinantes da saúde. Já a política de recursos humanos na área da saúde, formalizada e executada articuladamente pelas diferentes esferas de governo, tem entre seus objetivos a organização de um sistema de formação de recursos humanos em todos os níveis de ensino, inclusive de pós-graduação, além da elaboração de programas de permanente aperfeiçoamento de pessoal."),

    ("lei_8080_BR_90",
     "diretrizes terapêuticas do câncer incluirão a utilização de imunoterapia",
     "especialmente quanto à interoperabilidade para recebimento dos dados dos entes federativos",
     "O que a Lei nº 8.080 passou a exigir dos protocolos clínicos do câncer e que cuidado com dados ela impõe ao sistema de regulação assistencial?",
     "Os protocolos clínicos e as diretrizes terapêuticas do câncer devem incluir a utilização de imunoterapia quando esta se mostrar superior ou mais segura que as opções tradicionais, na forma do regulamento. Quanto ao sistema de regulação assistencial, a lei trata da interoperabilidade para recebimento dos dados dos entes federativos, garantidos o atendimento aos princípios e parâmetros da Lei nº 13.709/2018 quando aplicáveis."),

    # ── Código de Defesa do Consumidor ──────────────────────────────────────
    ("cdc_BR_1990",
     "O ônus da prova da veracidade e correção da informação ou comunicação publicitária",
     "É competente para a execução o juízo",
     "A quem o CDC atribui o ônus da prova sobre a publicidade e que juízo é competente para a execução da sentença?",
     "O ônus da prova da veracidade e correção da informação ou comunicação publicitária cabe a quem as patrocina. Para a execução, é competente o juízo da liquidação da sentença ou da ação condenatória, no caso de execução individual, e o da ação condenatória quando a execução for coletiva."),

    ("cdc_BR_1990",
     "manutenção de assistência jurídica, integral e gratuita para o consumidor carente",
     "poderá solicitar o concurso de órgãos e entidades de notória especialização técnico-científica",
     "Que instrumento o CDC lista primeiro para executar a Política Nacional das Relações de Consumo e a que recurso o Departamento Nacional de Defesa do Consumidor pode recorrer?",
     "Entre os instrumentos da Política Nacional das Relações de Consumo, o CDC lista a manutenção de assistência jurídica, integral e gratuita para o consumidor carente. Para alcançar seus objetivos, o Departamento Nacional de Defesa do Consumidor pode solicitar o concurso de órgãos e entidades de notória especialização técnico-científica."),

    # ── Código de Processo Ético-Profissional ───────────────────────────────
    ("resolucao_2306_CFM_2022",
     "tramitarão em sigilo processual",
     "quando for amigo íntimo ou inimigo de qualquer das partes",
     "Sob que regime de sigilo tramitam a sindicância e o processo ético-profissional e em que hipótese há suspeição do conselheiro?",
     "A sindicância e o processo ético-profissional nos CRMs e no CFM são regidos pelo CPEP e tramitam em sigilo processual, podendo correr em formato eletrônico nos termos de resolução específica. Há suspeição do conselheiro, entre outras hipóteses, quando ele for amigo íntimo ou inimigo de qualquer das partes ou de seus advogados."),

    ("resolucao_2306_CFM_2022",
     "a Corregedoria o remeterá à Coordenação Jurídica para exame de admissibilidade",
     "A testemunha fará a promessa de dizer a verdade",
     "O que acontece com o recurso assim que é recebido no CFM e o que se exige da testemunha ao depor?",
     "Recebido e autuado o recurso no CFM, a Corregedoria o remete à Coordenação Jurídica para exame de admissibilidade e, havendo preliminar processual arguida, emissão de Nota Técnica em cinco dias úteis. A testemunha, por sua vez, faz a promessa de dizer a verdade do que souber e for perguntado, devendo declarar seus dados e suas relações com as partes, e as testemunhas são inquiridas separadamente."),

    # ── ECA ─────────────────────────────────────────────────────────────────
    ("eca_BR_1990",
     "castigo físico: ação de natureza disciplinar ou punitiva aplicada com o uso da força física",
     "A remissão não implica necessariamente o reconhecimento ou comprovação da responsabilidade",
     "Como o ECA define castigo físico e o que a remissão não implica quanto à responsabilidade do adolescente?",
     "O ECA define castigo físico como ação de natureza disciplinar ou punitiva aplicada com o uso da força física sobre a criança ou o adolescente e que resulte em sofrimento ou lesão. A remissão, por sua vez, não implica necessariamente o reconhecimento ou a comprovação da responsabilidade, nem prevalece para efeito de antecedentes, podendo incluir a aplicação de medidas previstas em lei, exceto semiliberdade e internação."),

    ("eca_BR_1990",
     "o ensino obrigatório e gratuito é direito público subjetivo",
     "deixa de efetuar o cadastramento de crianças e de adolescentes em condições de serem adotadas",
     "Que natureza o ECA atribui ao ensino obrigatório e que conduta omissiva da autoridade ele pune quanto aos cadastros de adoção?",
     "O ECA afirma que o acesso ao ensino obrigatório e gratuito é direito público subjetivo, e que o não oferecimento ou a oferta irregular pelo poder público importa responsabilidade da autoridade competente. Em matéria de adoção, incorre nas mesmas penas a autoridade que deixa de efetuar o cadastramento de crianças e adolescentes em condições de serem adotados, de pessoas ou casais habilitados à adoção e de quem está em regime de acolhimento institucional ou familiar."),

    # ── Plano Brasileiro de IA ──────────────────────────────────────────────
    ("plano_brasileiro_ia_BR_2025",
     "conjunto de processos, métodos e técnicas para projetar, desenvolver, usar e implantar sistemas de IA",
     "instituído por meio do Decreto n.° 12.308/2024",
     "Como o Plano Brasileiro de IA define IA responsável e que colegiado federal assessora a Presidência na transformação digital?",
     "O plano define IA responsável como o conjunto de processos, métodos e técnicas para projetar, desenvolver, usar e implantar sistemas de IA que sejam éticos, confiáveis e benéficos para a sociedade, visando soluções justas, confiáveis e transparentes. O Comitê Interministerial para a Transformação Digital (CIT Digital) foi instituído pelo Decreto nº 12.308/2024 como órgão colegiado consultivo para assessorar o presidente da República em políticas de transformação digital."),

    # ── OMS e OCDE ──────────────────────────────────────────────────────────
    ("regulatory_considerations_ai_health_WHO_2024",
     "Validation processes and benchmarking should be carefully documented",
     "engagement and collaboration between developers, manufacturers, health-care practitioners",
     "O que a OMS espera que seja documentado nos processos de validação e que tipo de articulação entre partes ela aponta como capaz de melhorar a segurança do sistema de IA?",
     "A OMS espera que os processos de validação e de benchmarking sejam cuidadosamente documentados, incluindo as decisões de seleção de conjuntos de dados, padrões de referência, parâmetros e métricas que justifiquem tais processos. Além disso, aponta que o engajamento e a colaboração entre desenvolvedores, fabricantes, profissionais de saúde, pacientes, defensores de pacientes, formuladores de políticas e órgãos reguladores podem melhorar a segurança e a qualidade de um sistema de IA."),

    ("global_digital_health_strategy_WHO_2020_2027",
     "respects the privacy and security of patient health information",
     "digital health literacy, gender equality and women",
     "Que resguardo de informação a visão da estratégia global de saúde digital exige e o que o objetivo estratégico centrado nas pessoas busca promover?",
     "A visão da estratégia global de saúde digital pressupõe um sistema que respeite a privacidade e a segurança das informações de saúde do paciente, buscando ainda ampliar pesquisa, desenvolvimento, inovação e colaboração entre setores. O objetivo estratégico voltado a sistemas de saúde centrados nas pessoas promove letramento em saúde digital, igualdade de gênero e empoderamento das mulheres, além de abordagens inclusivas de adoção e gestão das tecnologias."),

    ("justice_data_governance_OECD_2024",
     "landlords are increasingly screening tenants using online eviction records",
     "Standards development processes may not be inclusive and support participation from all groups",
     "Que risco a OCDE ilustra ao tratar de dados de despejo em sistemas de gestão processual e que risco ela aponta nos processos de desenvolvimento de padrões?",
     "A OCDE ilustra que, nos Estados Unidos, proprietários vêm triando inquilinos com registros de despejo disponíveis nos sistemas de gestão processual dos tribunais, e que deixar de auditar ou curar a informação antes de abri-la pode afetar desproporcionalmente pessoas de baixa renda. Quanto aos padrões, aponta que os processos de desenvolvimento podem não ser inclusivos nem apoiar a participação de todos os grupos, produzindo padrões que não atendem às necessidades de todos os usuários do sistema de justiça."),

    # ── a 16ª, escrita em 30/09/2026 para fechar 75 multi_hop ────────────────
    # Alvo do artigo passou a ser 75/75/75 (decisão de 30/09), e o tipo estava em
    # 74. Documento escolhido por três critérios declarados antes de procurar
    # âncora: (a) não usado nas 15 anteriores; (b) grande — 457 trechos, então a
    # dispersão sai folgada; (c) evidência em INGLÊS, porque o multi_hop estava
    # 41 pt contra 33 en e o corte por idioma é do artigo. Capítulo 4 (leis de
    # proteção de dados, p. 35) contra capítulo 7 (explicabilidade, p. 123):
    # dispersão de ~311 trechos.
    ("ethics_governance_ai_health_WHO_2021",
     "Data protection laws also increasingly recognize that people have the right not to be subject to decisions guided solely by automated processes",
     "if a trade-off is to be made between transparency and accuracy, transparency should predominate",
     "Que direito a OMS afirma que as leis de proteção de dados vêm reconhecendo cada vez mais, e que ressalva ela faz ao argumento de que a transparência deve prevalecer sobre a acurácia?",
     "A OMS registra que as leis de proteção de dados protegem os direitos dos indivíduos e estabelecem obrigações para controladores e operadores, e que vêm reconhecendo cada vez mais o direito de não estar sujeito a decisões guiadas unicamente por processos automatizados — mais de 100 países já editaram leis desse tipo, entre elas o GDPR da União Europeia. Quanto à transparência, o documento observa que já se argumentou que, havendo um trade-off entre transparência e acurácia, a transparência deveria prevalecer, mas ressalva que essa exigência pode não ser possível nem mesmo desejável no contexto médico: embora muitas vezes se possa explicar por que um tratamento é a melhor opção para uma condição, não é sempre possível explicar como ele funciona ou qual seu mecanismo de ação, porque intervenções médicas às vezes são usadas antes de seu modo de ação ser compreendido."),
]


_PT = (" de ", " que ", " nao ", " sao ", " para ", " com ", " dos ", " uma ")
_EN = (" the ", " of ", " and ", " shall ", " this ", " is ", " which ", " to ")


def detectar_idioma(texto: str) -> str:
    """'pt' ou 'en' por contagem de marcadores — mesma regra de
    `eval/lib/retrievers.detect_lang`, para o corte de idioma bater com o que o
    BM25 bilíngue usa para escolher stopwords e stemmer."""
    t = " " + normalizar(texto) + " "
    return "en" if sum(t.count(w) for w in _EN) > sum(t.count(w) for w in _PT) else "pt"


def normalizar(t: str) -> str:
    t = (t or "").replace("­", "")
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    trechos, temas = [], {}
    with CORPUS.open(encoding="utf-8") as fh:
        for linha in fh:
            if not linha.strip():
                continue
            r = json.loads(linha)
            m = r["metadata"]
            trechos.append(normalizar(r["text"]))
            temas.setdefault(m["document_id"], m.get("theme"))

    registros, falhas, juntas = [], [], []
    for i, (doc, aa, ab, pergunta, resposta) in enumerate(ESPEC, start=1):
        try:
            # JANELA LARGA, E COM `antes`. Com os 170 caracteres padrão, a
            # citação começava NA âncora e terminava antes do que a resposta
            # afirmava — o sujeito da oração ("o Departamento Nacional...",
            # "incorre em pena de multa a autoridade que...") ficava atrás, e a
            # consequência ("que resulte em sofrimento ou lesão") ficava à
            # frente. Na primeira leitura 11 das 15 respostas afirmavam algo
            # fora da própria citação. Citação maior não custa nada na métrica
            # — o qrels aponta o trecho, não o recorte — e é o que faz a
            # evidência sustentar mesmo a resposta.
            ev = [citar(doc, aa, janela=560, antes=120),
                  citar(doc, ab, janela=560, antes=120)]
        except AncoraRuim as erro:
            falhas.append((i, str(erro)))
            continue

        posicoes = []
        for e in ev:
            alvo = normalizar(e)
            achados = [j for j, t in enumerate(trechos) if alvo in t]
            posicoes.append(min(achados) if achados else None)
        if None in posicoes:
            falhas.append((i, f"{doc}: citação extraída não localizada no corpus"))
            continue
        dispersao = max(posicoes) - min(posicoes)
        if dispersao < DISPERSAO_MINIMA:
            juntas.append((i, doc, dispersao))
            continue

        registros.append({
            "n": f"m{len(registros)+1:03d}",
            "question": pergunta,
            "answer": resposta,
            "evidencia": ev,
            "n_evidencias": 2,
            "documento": doc,
            "forma": None,
            "janela": None,
            "passagem_texto": None,
            "question_type": "multi_hop",
            "difficulty": "hard" if dispersao > 20 else "medium",
            "theme": temas.get(doc, ""),
            # Idioma DETECTADO no texto da citação. O metadata do corpus não tem
            # `forma` — o campo existe só nas 91, que o trouxeram de fora. Derivar
            # dele aqui devolvia "pt" para os documentos da OMS e da OCDE, e o
            # corte PT×EN do artigo mediria o contrário do que diz.
            "source_lang": detectar_idioma(" ".join(ev)),
            "dispersao_trechos": dispersao,
            "origem": "multihop_dispersas",
        })

    print(f"multi-hop montadas: {len(registros)} de {len(ESPEC)}")
    if falhas:
        print(f"\nâncoras a corrigir: {len(falhas)}")
        for i, e in falhas:
            print(f"   [{i}] {e}")
    if juntas:
        print(f"\n✗ dispersão abaixo de {DISPERSAO_MINIMA} — não são multi-hop:")
        for i, d, x in juntas:
            print(f"   [{i}] {d}: dispersão {x}")

    if registros:
        ds = [r["dispersao_trechos"] for r in registros]
        print(f"\ndispersão: mín {min(ds)}  mediana {sorted(ds)[len(ds)//2]}  máx {max(ds)}")
        print(f"documentos : {len(Counter(r['documento'] for r in registros))}")
        print(f"idioma     : {dict(Counter(r['source_lang'] for r in registros))}")
        print(f"dificuldade: {dict(Counter(r['difficulty'] for r in registros))}")
        print("\n── citações extraídas, para leitura ──")
        for r in registros:
            print(f"\n[{r['n']}] {r['question']}")
            for j, e in enumerate(r["evidencia"], start=1):
                print(f"   ({j}) {e[:190]}")

    if args.aplicar and not falhas and not juntas:
        SAIDA.write_text(json.dumps(registros, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        print(f"\n✓ {SAIDA}")
    elif args.aplicar:
        print("\n✗ não gravei: corrija as âncoras e a dispersão primeiro.")
    else:
        print("\n(relatório apenas — use --aplicar para gravar)")


if __name__ == "__main__":
    main()
