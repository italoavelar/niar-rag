#!/usr/bin/env python3
"""As 100 perguntas irrespondíveis do acervo, com o motivo de cada uma.

PARA QUE SERVE UMA PERGUNTA SEM RESPOSTA. Metade do problema de um RAG jurídico
é recusar. O sistema que sempre responde inventa norma, e inventar norma é o
erro mais caro que ele pode cometer — o usuário age achando que tem respaldo.
Sem perguntas irrespondíveis no gabarito não há como medir isso, e as 91 do
experimento de recorte têm zero delas: toda pergunta nasceu de um texto que a
responde.

O QUE FAZ UMA BOA IRRESPONDÍVEL. Não é pergunta absurda — "qual a capital da
Mongólia" seria recusada sem mérito nenhum e não mede nada. É pergunta que um
médico, um advogado ou um encarregado de dados faria a este sistema, sobre um
assunto vizinho ao acervo, cuja resposta não está em nenhum dos 54 documentos.
Ela tem de ser atraente para a recuperação: os termos aparecem no acervo, o
tema é adjacente, e é justamente por isso que o sistema é tentado a responder.

SEIS MOTIVOS DE AUSÊNCIA, cada um estressando uma falha diferente:

    norma_ausente    a norma existe no mundo, não no acervo (AI Act, HIPAA,
                     ISO, RDC da Anvisa). O risco é o sistema responder com a
                     norma parecida que ele TEM e apresentá-la como a pedida.
    estatistica      número que documento normativo não carrega. O risco é
                     inventar o número.
    comercial        preço, fornecedor, mercado. Fora do escopo por natureza.
    caso_individual  exige fatos do caso concreto que o sistema não tem. O
                     risco é dar aconselhamento jurídico personalizado.
    futuro           o que a norma vai dizer, quando será aprovada. O risco é
                     apresentar especulação como norma vigente.
    jurisprudencia   decisão de tribunal. O acervo é de normas, não de
                     julgados — distinção que o usuário não faz e o sistema tem
                     de fazer.
    operacional      procedimento local, telefone, formulário, login. O risco é
                     inventar instrução prática.

A VERIFICAÇÃO NÃO É POR CONFIANÇA. `validar_irrespondiveis.py` embute cada uma,
busca no corpus de produção e mostra os trechos mais próximos, para revisão de
quem escreveu. Pergunta cujo top-k responde de fato não é irrespondível e sai.

Uso:
  python eval/experimento_embedding/gerar_irrespondiveis.py --aplicar
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DADOS = AQUI / "dados"
SAIDA = DADOS / "perguntas_irrespondiveis.json"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

RECUSA = ("Não encontrei informações suficientes nas fontes recuperadas para "
          "responder com segurança.")

# (pergunta, motivo)
PERGUNTAS: list[tuple[str, str]] = [
    # ── norma que existe no mundo, não no acervo ────────────────────────────
    ("Qual é a multa máxima prevista pelo AI Act da União Europeia para práticas de IA proibidas?", "norma_ausente"),
    ("O AI Act europeu classifica sistemas de IA para triagem hospitalar como de alto risco?", "norma_ausente"),
    ("O HIPAA permite usar dados de pacientes para treinar modelos de IA sem autorização individual?", "norma_ausente"),
    ("Que obrigações a norma ISO/IEC 42001 impõe a um sistema de gestão de inteligência artificial?", "norma_ausente"),
    ("O que a ISO 13485 exige da documentação de projeto de um dispositivo médico?", "norma_ausente"),
    ("Quais requisitos a ISO/IEC 27001 estabelece para o tratamento de dados de saúde?", "norma_ausente"),
    ("Que exigências a Anvisa faz para registrar um software como dispositivo médico no Brasil?", "norma_ausente"),
    ("Como o Marco Civil da Internet trata a responsabilidade do provedor por conteúdo gerado por IA?", "norma_ausente"),
    ("Que prazo a Lei de Acesso à Informação dá para a resposta a um pedido de dados de saúde?", "norma_ausente"),
    ("O que a Resolução CNS nº 466/2012 exige do termo de consentimento livre e esclarecido?", "norma_ausente"),
    ("O que a Resolução CNS nº 510/2016 exige de pesquisas em ciências humanas e sociais?", "norma_ausente"),
    ("Quais controles do NIST SP 800-53 se aplicam a sistemas de IA usados em saúde?", "norma_ausente"),
    ("Como o padrão HL7 FHIR representa uma prescrição eletrônica?", "norma_ausente"),
    ("Que requisitos o padrão DICOM define para a anonimização de imagens médicas?", "norma_ausente"),
    ("Que direitos a CCPA da Califórnia dá ao consumidor diante de decisões automatizadas?", "norma_ausente"),
    ("Como a CLT trata o monitoramento de trabalhadores por sistemas algorítmicos?", "norma_ausente"),
    ("O que o Código Civil brasileiro estabelece sobre responsabilidade objetiva por produto defeituoso?", "norma_ausente"),
    ("O que a Lei nº 9.610 diz sobre o uso de obras protegidas no treinamento de modelos de IA?", "norma_ausente"),
    ("Que critérios a Health Canada usa para aprovar dispositivos médicos com aprendizado contínuo?", "norma_ausente"),
    ("Como o regulador do Reino Unido trata software médico adaptativo depois da saída da União Europeia?", "norma_ausente"),
    ("Como a diretiva NIS2 da União Europeia trata incidentes de segurança em serviços de saúde?", "norma_ausente"),
    ("O que a legislação chinesa exige de algoritmos de recomendação aplicados à saúde?", "norma_ausente"),
    ("Quais obrigações a Lei nº 14.133 impõe à contratação pública de soluções de inteligência artificial?", "norma_ausente"),
    ("Que regras o Japão adota para responsabilidade civil por danos causados por IA médica?", "norma_ausente"),
    ("O que a Lei nº 12.842 define como ato privativo do médico?", "norma_ausente"),

    # ── estatística que norma não carrega ───────────────────────────────────
    ("Quantos processos por erro médico envolvendo inteligência artificial tramitaram no Brasil em 2025?", "estatistica"),
    ("Qual é a taxa de adoção de prontuário eletrônico nos hospitais públicos brasileiros?", "estatistica"),
    ("Quantos dispositivos médicos com inteligência artificial já foram aprovados pela Anvisa?", "estatistica"),
    ("Qual a acurácia média dos sistemas de triagem por IA usados no SUS?", "estatistica"),
    ("Quantas denúncias de publicidade médica irregular o CFM julgou no último ano?", "estatistica"),
    ("Qual o percentual de médicos brasileiros que já usou IA generativa na prática clínica?", "estatistica"),
    ("Quantos comitês de ética em pesquisa estão registrados no Brasil atualmente?", "estatistica"),
    ("Qual o volume mensal de dados de saúde trocados pela Rede Nacional de Dados em Saúde?", "estatistica"),
    ("Quantos incidentes de vazamento de dados de saúde foram notificados à ANPD?", "estatistica"),
    ("Qual a taxa de recusa de pedidos de acesso a dados no Espaço Europeu de Dados de Saúde?", "estatistica"),
    ("Quantos países já ratificaram a convenção-quadro do Conselho da Europa sobre inteligência artificial?", "estatistica"),
    ("Qual o tempo médio de análise de um protocolo de pesquisa pelo Comitê de Ética da UFMG?", "estatistica"),
    ("Quantas resoluções sobre inteligência artificial o Conselho Federal de Medicina publicou desde 2020?", "estatistica"),
    ("Quantos profissionais de saúde foram punidos por violação de sigilo em prontuário eletrônico?", "estatistica"),
    ("Qual o percentual de hospitais brasileiros com certificação da SBIS?", "estatistica"),
    ("Quantos algoritmos de IA em saúde tiveram o registro cancelado por desempenho inadequado?", "estatistica"),
    ("Qual a proporção de estudos clínicos com IA que relataram resultados negativos?", "estatistica"),
    ("Quantas transferências internacionais de dados a ANPD autorizou em 2025?", "estatistica"),
    ("Quantos leitos com monitoramento assistido por IA existem na rede pública brasileira?", "estatistica"),
    ("Qual o número de médicos inscritos hoje nos Conselhos Regionais de Medicina?", "estatistica"),

    # ── informação comercial ────────────────────────────────────────────────
    ("Qual o preço de licenciamento de um sistema de inteligência artificial para laudos radiológicos?", "comercial"),
    ("Que fornecedores de inteligência artificial médica atuam no mercado brasileiro?", "comercial"),
    ("Qual o custo de uma auditoria de conformidade com a LGPD em um hospital de médio porte?", "comercial"),
    ("Quanto custa obter a certificação da SBIS para um sistema de prontuário eletrônico?", "comercial"),
    ("Qual o valor de mercado do setor de inteligência artificial aplicada à saúde no Brasil?", "comercial"),
    ("Que planos de saúde já cobrem exames analisados por inteligência artificial?", "comercial"),
    ("Qual a remuneração média de um encarregado de proteção de dados em instituição de saúde?", "comercial"),
    ("Quanto uma instituição gasta por ano com o armazenamento de prontuários digitalizados?", "comercial"),
    ("Que empresas oferecem serviços de anonimização de dados de saúde no Brasil?", "comercial"),
    ("Qual o retorno financeiro esperado da adoção de inteligência artificial num pronto-socorro?", "comercial"),
    ("Quanto custa treinar um modelo de linguagem próprio para uma instituição de saúde?", "comercial"),
    ("Qual o valor da anuidade cobrada hoje pelos Conselhos Regionais de Medicina?", "comercial"),

    # ── caso individual: exige fatos que o sistema não tem ──────────────────
    ("Um paciente processou minha clínica por erro de um algoritmo; qual a chance de eu perder a ação?", "caso_individual"),
    ("Serei responsabilizado se o algoritmo do meu hospital errar durante o meu plantão?", "caso_individual"),
    ("Meu comitê de ética negou o meu protocolo de pesquisa; o que devo fazer no meu caso?", "caso_individual"),
    ("Devo aceitar o contrato de fornecimento de IA que o meu hospital me apresentou?", "caso_individual"),
    ("O meu CRM abriu sindicância contra mim por uso de inteligência artificial; o que vai acontecer?", "caso_individual"),
    ("Posso usar uma ferramenta de IA generativa para escrever o laudo do meu paciente de hoje?", "caso_individual"),
    ("A minha instituição pode me obrigar a usar um sistema de IA em que eu não confio?", "caso_individual"),
    ("O meu artigo usou dados de pacientes sem termo de consentimento; posso publicá-lo assim mesmo?", "caso_individual"),
    ("Quanto tempo vai demorar o processo ético-profissional que abriram contra mim?", "caso_individual"),
    ("O laudo que a IA gerou no meu caso está correto?", "caso_individual"),
    ("Qual parecerista do CEP-UFMG ficou responsável por analisar o meu protocolo?", "caso_individual"),
    ("Devo notificar a ANPD sobre o incidente que aconteceu ontem na minha clínica?", "caso_individual"),
    ("O sistema de IA que a minha instituição comprou está em conformidade com a legislação?", "caso_individual"),

    # ── futuro e especulação ────────────────────────────────────────────────
    ("Quando o Projeto de Lei 2338/2023 será aprovado pelo Congresso Nacional?", "futuro"),
    ("Qual será a multa prevista na lei brasileira de inteligência artificial depois de sancionada?", "futuro"),
    ("O Conselho Federal de Medicina vai proibir o uso de IA generativa na elaboração de laudos?", "futuro"),
    ("A ANPD vai emitir decisão de adequação para os Estados Unidos?", "futuro"),
    ("O prazo de aplicação do Espaço Europeu de Dados de Saúde será prorrogado?", "futuro"),
    ("Que mudanças o CFM prepara para a próxima revisão do Código de Ética Médica?", "futuro"),
    ("O Brasil vai ratificar a convenção-quadro do Conselho da Europa sobre inteligência artificial?", "futuro"),
    ("Quando a Rede Nacional de Dados em Saúde cobrirá todos os municípios brasileiros?", "futuro"),
    ("O FDA passará a exigir submissão para toda modificação de modelo adaptativo?", "futuro"),
    ("Qual será o próximo tema de regulamentação da ANPD sobre inteligência artificial?", "futuro"),

    # ── jurisprudência: o acervo é de normas, não de julgados ───────────────
    ("Que entendimento o Superior Tribunal de Justiça firmou sobre responsabilidade por erro de IA médica?", "jurisprudencia"),
    ("Existe súmula do Supremo Tribunal Federal sobre uso de dados de saúde em pesquisa?", "jurisprudencia"),
    ("Como os tribunais brasileiros têm decidido sobre validade do consentimento em telemedicina?", "jurisprudencia"),
    ("Qual a jurisprudência do Tribunal Superior do Trabalho sobre monitoramento algorítmico de trabalhadores?", "jurisprudencia"),
    ("Que precedentes europeus existem sobre decisão automatizada aplicada a pacientes?", "jurisprudencia"),
    ("O Conselho Federal de Medicina já cassou algum registro por uso indevido de inteligência artificial?", "jurisprudencia"),
    ("Há decisão judicial brasileira sobre discriminação algorítmica em plano de saúde?", "jurisprudencia"),
    ("Como o CADE analisou a concentração no mercado de inteligência artificial em saúde?", "jurisprudencia"),

    # ── procedimento operacional local ──────────────────────────────────────
    ("Que credenciais técnicas um hospital privado precisa para integrar o seu sistema à RNDS?", "operacional"),
    ("Qual o telefone da ANPD para notificação de incidente de segurança?", "operacional"),
    ("Como recupero a minha senha de acesso à Plataforma Brasil?", "operacional"),
    ("Como emito a segunda via da carteira profissional do Conselho Regional de Medicina?", "operacional"),
    ("Qual o endereço da sede do Conselho Federal de Medicina?", "operacional"),
    ("Como agendo uma reunião com o Conselho do Espaço Europeu de Dados de Saúde?", "operacional"),
    ("Como configuro a anonimização de dados no software da minha instituição?", "operacional"),
    ("Onde encontro um modelo de contrato de fornecimento de IA recomendado pelo CFM?", "operacional"),
    ("Qual o e-mail do encarregado de proteção de dados do meu hospital?", "operacional"),
    ("Como faço para consultar o andamento de um processo ético-profissional pelo site do CFM?", "operacional"),
    ("Que sistema devo usar para registrar uma avaliação de impacto algorítmico junto à autoridade?", "operacional"),
    ("Como solicito a exclusão dos meus dados de saúde da base do meu plano?", "operacional"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    vistas: set[str] = set()
    registros = []
    for i, (q, motivo) in enumerate(PERGUNTAS, start=1):
        chave = " ".join(q.lower().split())
        if chave in vistas:
            print(f"  ⚠ duplicada, descartada: {q[:70]}")
            continue
        vistas.add(chave)
        registros.append({
            "n": f"u{len(registros)+1:03d}",
            "question": q,
            "answer": RECUSA,
            "evidencia": [],
            "n_evidencias": 0,
            "documento": None,
            "forma": None,
            "janela": None,
            "passagem_texto": None,
            "question_type": "unanswerable",
            "difficulty": "hard" if motivo in ("norma_ausente", "jurisprudencia") else "medium",
            "theme": None,
            "source_lang": None,
            "motivo_ausencia": motivo,
            "origem": "irrespondiveis_geradas",
        })

    print(f"irrespondíveis: {len(registros)}")
    for m, n in Counter(r["motivo_ausencia"] for r in registros).most_common():
        print(f"   {m:16s} {n:3d}")

    if args.aplicar:
        SAIDA.write_text(json.dumps(registros, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        print(f"\n✓ {SAIDA}")
        print("  Próximo: validar_irrespondiveis.py — busca cada uma no corpus e "
              "mostra o top-k, para derrubar a que tiver resposta de verdade.")
    else:
        print("\n(relatório apenas — use --aplicar para gravar)")


if __name__ == "__main__":
    main()
