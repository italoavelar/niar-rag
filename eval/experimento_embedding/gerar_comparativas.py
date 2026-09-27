#!/usr/bin/env python3
"""As comparativas do acervo: perguntas que só se respondem cruzando dois documentos.

O QUE UMA COMPARATIVA COBRA DA RECUPERAÇÃO, e nenhum outro tipo cobra. Factual e
multi-hop vivem dentro de um documento: basta o recuperador acertar o documento
certo e trazer os trechos certos dele. Comparativa obriga a trazer **dois
documentos diferentes** no mesmo top-k. É o caso em que o ranking pode se
entupir com um só lado — cinco trechos ótimos da LGPD e nenhum do RGPD — e a
resposta fica impossível mesmo com precisão alta. Nenhuma das 91 do experimento
de recorte testa isso: cada uma nasceu de uma passagem de um documento só.

AS CITAÇÕES NÃO SÃO TRANSCRITAS. Cada uma é extraída do corpus por
`lib_citacao.citar`, a partir de uma âncora curta. Transcrever 182 citações à mão
erraria, e erraria em silêncio — a citação quase certa não casa, a pergunta é
rejeitada e alguém procura o acento trocado. Aqui o texto sai do corpus por
construção, byte a byte.

A RESPOSTA DECLARA OS DOIS LADOS e, quando cabe, o contraste entre eles. Sem
isso a pergunta não é comparativa de verdade: seria duas factuais coladas.

Uso:
  python eval/experimento_embedding/gerar_comparativas.py
  python eval/experimento_embedding/gerar_comparativas.py --aplicar
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
SAIDA = DADOS / "perguntas_comparativas.json"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

from lib_citacao import AncoraRuim, citar  # noqa: E402

# (doc_a, âncora_a, doc_b, âncora_b, pergunta, resposta, dificuldade)
ESPEC: list[tuple[str, str, str, str, str, str, str]] = [

    # ── Conselhos de Medicina e processo ético ──────────────────────────────
    ("lei_3268_BR_1957", "cassação do exercício profissional, ad referendum",
     "decreto_44045_BR_1958", "a) advertência confidencial, em aviso reservado",
     "Quais são as penas disciplinares do exercício médico e o que a lei exige quanto à gradação delas?",
     "A Lei nº 3.268/1957 e o Decreto nº 44.045/1958 listam as mesmas cinco penas — advertência confidencial, censura confidencial, censura pública, suspensão até 30 dias e cassação do exercício profissional. A lei acrescenta que a cassação é ad referendum do Conselho Federal e que, salvo gravidade manifesta que exija a pena mais grave de imediato, a imposição obedece à gradação do artigo.",
     "hard"),

    ("lei_6681_BR_1979", "participarem de eleições nos Conselhos",
     "lei_3268_BR_1957", "de 21 (vinte e um), quando excedido",
     "Como se compõem os Conselhos Regionais de Medicina e que restrição eleitoral recai sobre os médicos militares?",
     "A Lei nº 3.268/1957 escalona a composição pelo número de inscritos, chegando a 21 conselheiros quando esse número é excedido, eleitos em escrutínio secreto com exceção de um escolhido pela Associação Médica. A Lei nº 6.681/1979 veda aos médicos, cirurgiões-dentistas e farmacêuticos militares participar das eleições nos Conselhos em que estiverem inscritos, seja como candidatos, seja como eleitores.",
     "hard"),

    ("lei_6681_BR_1979", "não estão sujeitos à ação disciplinar",
     "resolucao_2306_CFM_2022", "A competência para instaurar sindicância",
     "Qual CRM é competente para apurar falta ética e o que acontece quando o médico é militar?",
     "Pelo Código de Processo Ético-Profissional, a competência para instaurar sindicância e, se for o caso, o PEP é do CRM onde o fato punível ocorreu, ainda que o médico não tenha inscrição naquela circunscrição ou já tenha se transferido. A Lei nº 6.681/1979 abre exceção: no exercício de atividades técnico-profissionais decorrentes da condição militar, esses profissionais não se sujeitam à ação disciplinar dos Conselhos Regionais, e sim à da Força Singular.",
     "hard"),

    ("resolucao_2424_CFM_2025", "poderá delegar às respectivas Corregedorias",
     "resolucao_2306_CFM_2022", "A suspeição poderá ser alegada, no prazo de 15",
     "Quem designa instrutor e relator no processo ético-profissional e em que prazo se alega suspeição?",
     "A Resolução CFM nº 2.424/2025 alterou o CPEP para permitir que a Presidência dos Conselhos delegue às respectivas Corregedorias a competência de designar sindicante, instrutor e relator, além de lavrar portarias. O CPEP estabelece que a suspeição seja alegada no prazo de 15 dias contados do conhecimento do fato, em petição específica.",
     "medium"),

    ("codigo_etica_medica_CFM_2019", "Revelar fato de que tenha conhecimento em virtude do exercício",
     "resolucao_2306_CFM_2022", "A comissão de ética médica dos estabelecimentos",
     "O que o Código de Ética Médica veda quanto ao sigilo e a quem cabe comunicar denúncias éticas nos estabelecimentos de saúde?",
     "O Código de Ética Médica veda ao médico revelar fato de que tenha conhecimento em virtude do exercício da profissão, salvo por motivo justo, dever legal ou consentimento escrito do paciente. Pelo CPEP, a comissão de ética médica do estabelecimento de saúde deve encaminhar ao CRM as denúncias de natureza ética de que tiver ciência; não havendo comissão, cabe ao diretor clínico ou técnico fazer a comunicação.",
     "hard"),

    ("lei_3268_BR_1957", "fixar e alterar o valor da anuidade única",
     "decreto_44045_BR_1958", "acréscimo de 20% (vinte por cento)",
     "Quem fixa a anuidade dos Conselhos de Medicina e qual a consequência de pagá-la fora do prazo?",
     "A Lei nº 3.268/1957 atribui ao Conselho Federal fixar e alterar o valor da anuidade única cobrada aos inscritos nos Conselhos Regionais. O Decreto nº 44.045/1958 determina que o pagamento fora do prazo seja efetuado com acréscimo de 20% da importância fixada.",
     "medium"),

    # ── Prontuário e registro eletrônico ────────────────────────────────────
    ("resolucao_1638_CFM_2002", "documento único constituído de um conjunto de informações",
     "lei_13787_BR_2018", "20 (vinte) anos a partir do último registro",
     "Como o CFM define prontuário médico e por quanto tempo a lei exige que ele seja guardado?",
     "A Resolução CFM nº 1.638/2002 define prontuário médico como documento único constituído de um conjunto de informações, sinais e imagens registradas, de caráter legal, sigiloso e científico, que possibilita a comunicação entre a equipe e a continuidade da assistência. A Lei nº 13.787/2018 permite eliminar os prontuários, em papel ou digitalizados, depois de 20 anos contados do último registro, admitindo prazos diferenciados em regulamento.",
     "hard"),

    ("resolucao_1821_CFM_2007", "Manual de Certificação para Sistemas de Registro Eletrônico",
     "certificacao_SBIS_", "selo",
     "Que papel a certificação cumpre nos sistemas de registro eletrônico em saúde e na avaliação de soluções de IA?",
     "A Resolução CFM nº 1.821/2007 aprovou o Manual de Certificação para Sistemas de Registro Eletrônico em Saúde, disponibilizado pelo CFM e pela SBIS. No caso da inteligência artificial, a proposta de certificação da SBIS oferece um selo que ajuda empresas e instituições a demonstrar conformidade com boas práticas, promovendo harmonização regulatória.",
     "hard"),

    ("rnds_MS_", "plataforma oficial de interoperabilidade",
     "european_health_data_space_EU_2025", "Cada Estado-Membro designa um ponto de contacto nacional",
     "Que estrutura o Brasil e a União Europeia criaram para a troca de dados de saúde entre sistemas?",
     "No Brasil, a Rede Nacional de Dados em Saúde é a plataforma oficial de interoperabilidade do Ministério da Saúde, para conectar sistemas de saúde em todo o país. Na União Europeia, cada Estado-Membro designa um ponto de contacto nacional para utilização secundária, que funciona como portal organizacional e técnico responsável por disponibilizar dados de saúde eletrónicos nesse contexto.",
     "hard"),

    # ── Pesquisa com seres humanos ──────────────────────────────────────────
    ("regulamento_interno_CEP_2024", "Plataforma Brasil que é uma base nacional e unificada",
     "lei_14874_BR_2024", "pessoa responsável pela condução da pesquisa em instituição",
     "Por onde tramita um protocolo de pesquisa com seres humanos e como a lei define o pesquisador responsável?",
     "O regulamento do CEP-UFMG determina que o protocolo tramite pela Plataforma Brasil, base nacional e unificada de registros de pesquisas envolvendo seres humanos para todo o sistema CEP/CONEP. A Lei nº 14.874/2024 define o pesquisador ou investigador como a pessoa responsável pela condução da pesquisa na instituição ou centro de pesquisa e corresponsável pela integridade e pelo bem-estar dos participantes.",
     "hard"),

    ("resolucao_UFMG_01_2022", "cartas de aprovação dos projetos de pesquisa permanecerão sob guarda",
     "regulamento_interno_CEP_2024", "retirado do sistema no prazo de um ano",
     "O que acontece com os protocolos de pesquisa sem resposta e o que fica sob guarda dos comitês depois do prazo?",
     "O regulamento do CEP-UFMG determina que o processo seja retirado do sistema no prazo de um ano em caso de não resposta à pendência. A Resolução UFMG nº 01/2022 estabelece que, após o período mínimo previsto, apenas as cartas de aprovação dos projetos de pesquisa permaneçam sob guarda dos Comitês.",
     "hard"),

    ("tcud_CEP_2015", "exclusivamente para os fins científicos",
     "lei_14874_BR_2024", "Termo de Transferência de Material Biológico",
     "Que compromissos formais cercam o uso de dados e de material biológico em pesquisa?",
     "O termo de ciência sobre utilização de dados condiciona a cessão ao cumprimento da Resolução 466/12 e suas complementares, comprometendo o pesquisador a usar os dados exclusivamente para fins científicos, mantendo sigilo e garantindo a não utilização em prejuízo das pessoas ou comunidades. A Lei nº 14.874/2024 exige, para a transferência de material biológico humano, a celebração de Termo de Transferência de Material Biológico e a comprovação de aprovação ética e regulatória.",
     "hard"),

    # ── Proteção de dados: avaliação de impacto e porte do agente ───────────
    ("dpia_ICO_", "fine of up to £8.7 million",
     "gdpr_regulation_EU_2016", "A consultation of the supervisory authority should also take place",
     "Que consequência há em não fazer a avaliação de impacto quando ela é exigida, e quando se consulta a autoridade de controlo?",
     "O guia da ICO alerta que, sob o RGPD do Reino Unido, deixar de realizar a DPIA quando exigida expõe a organização a medidas de execução, incluindo multa de até 8,7 milhões de libras ou 2% do faturamento global anual, o que for maior. O RGPD prevê ainda que a autoridade de controlo seja consultada durante a preparação de medida legislativa ou regulamentar que preveja tratamento de dados pessoais.",
     "hard"),

    ("dpia_ICO_", "at a point when there is a realistic opportunity to influence",
     "projeto_lei_2338_BR_2023", "periodicidade de atualização das avaliações de impacto algorítmico",
     "Em que momento a avaliação de impacto deve ser feita e quando precisa ser atualizada?",
     "A ICO exige que a DPIA seja feita num momento em que ainda haja oportunidade real de influenciar os planos — avaliação depois da decisão tomada não cumpre a função. O PL 2338/2023 trata do outro extremo do ciclo: cabe à autoridade competente definir parâmetros de periodicidade de atualização das avaliações de impacto algorítmico, que devem ocorrer ao menos quando houver alterações significativas nos sistemas.",
     "hard"),

    ("resolucao_2_ANPD_2024", "agentes de tratamento de pequeno porte: microempres",
     "lgpd_BR_2018", "anonimização: utilização de meios técnicos razoáveis e disponíveis",
     "Quem é agente de tratamento de pequeno porte e como a LGPD define anonimização?",
     "A Resolução CD/ANPD nº 2 define agentes de tratamento de pequeno porte a partir de categorias como microempresas, aplicando-lhes regime próprio de cumprimento da LGPD. A LGPD define anonimização como a utilização de meios técnicos razoáveis e disponíveis no momento do tratamento, por meio dos quais um dado perde a possibilidade de associação a um indivíduo.",
     "hard"),

    ("resolucao_2_ANPD_2024", "podem cumprir a obrigação de elaboração e manutenção de registro",
     "resolucao_19_ANPD_2024", "o Importador deve manter registro de Solicitações de Acesso",
     "Que obrigações de registro a ANPD impõe ao agente de pequeno porte e a quem recebe dados no exterior?",
     "Para o agente de tratamento de pequeno porte, a ANPD admite forma simplificada de cumprir a obrigação de elaborar e manter o registro das operações de tratamento. Já na transferência internacional, o Importador deve manter registro das Solicitações de Acesso com data, solicitante, finalidade, tipo de dados, número de solicitações recebidas e medidas legais adotadas.",
     "hard"),

    # ── Dispositivos médicos com aprendizado de máquina ─────────────────────
    ("imdrf_ml_medical_devices_terms_2022", "Training that leads to change of an MLMD with each exposure to data",
     "ai_ml_discussion_paper_FDA_2021", "We define a “locked” algorithm",
     "Como se distinguem algoritmo travado e aprendizado contínuo num dispositivo médico?",
     "O IMDRF define aprendizado contínuo como o treinamento que leva à mudança do dispositivo a cada exposição a dados, de forma permanente durante a fase de operação. O FDA define no outro extremo o algoritmo travado: aquele que devolve o mesmo resultado cada vez que a mesma entrada lhe é aplicada e não muda com o uso.",
     "hard"),

    ("imdrf_samd_risk_framework_2014", "Critical situation or condition",
     "imdrf_samd_clinical_evaluation_2017", "independent review of clinical evidence",
     "Como o IMDRF categoriza o risco de um software como dispositivo médico e o que isso implica para a revisão independente?",
     "No quadro do IMDRF, o SaMD que informa decisão em situação ou condição crítica é Categoria IV, considerada de impacto muito alto — a faixa superior da escala de risco. Na avaliação clínica, e sujeito às leis de cada jurisdição, a revisão independente da evidência clínica de certos SaMD de baixo risco pode ser menos importante, podendo o fabricante autodeclarar a adequação da evidência.",
     "hard"),

    ("transparency_ml_devices_FDA_2024", "known gaps in the data characterization",
     "multimodal_models_guidance_WHO_2024", "no major foundation model developer is close to providing adequate transparency",
     "O que se espera que um dispositivo com aprendizado de máquina revele sobre seus dados, e como isso se compara ao que a OMS observa nos modelos de fundação?",
     "O FDA orienta que se informem as lacunas conhecidas na caracterização dos dados, incluindo populações de pacientes mal representadas nos conjuntos de treinamento ou clínicos, que por isso ficam sob risco de viés. A OMS constata o oposto no mercado de modelos de fundação: nenhum grande desenvolvedor chega perto de oferecer transparência adequada, o que revela falta fundamental de transparência na indústria.",
     "hard"),

    ("predetermined_change_control_FDA_2023", "Focused and Bounded",
     "ai_ml_discussion_paper_FDA_2021", "An ACP is a description of the set of specific methods",
     "O que caracteriza um plano predeterminado de controle de mudanças e o que um protocolo de alteração de algoritmo descreve?",
     "O primeiro princípio orientador do PCCP é ser focado e delimitado: o plano descreve mudanças específicas que o fabricante pretende implementar, limitadas a modificações dentro do uso pretendido do dispositivo original. O ACP, na proposta do FDA, é a descrição do conjunto de métodos específicos que o fabricante tem para alcançar e controlar adequadamente os riscos dos tipos de modificação previstos no SPS.",
     "hard"),

    ("transparency_ml_devices_FDA_2024", "human-centered design",
     "ai_device_software_guidance_FDA_2025", "Appendix B (Transparency Design Considerations)",
     "Que abordagem de projeto o FDA associa à transparência de dispositivos com aprendizado de máquina?",
     "Os princípios orientadores de transparência apontam o desenho centrado no humano — processo iterativo que trata a experiência completa do usuário e envolve as partes relevantes ao longo do desenvolvimento. A guidance de 2025 materializa isso num apêndice de considerações de projeto para transparência, que orienta o entendimento das indicações de uso do dispositivo e do cartão de modelo.",
     "hard"),

    ("imdrf_ml_medical_devices_terms_2022", "A product must first meet the definition of a medical device",
     "imdrf_good_ml_practice_2025", "Training datasets are independent of test sets",
     "O que precede a classificação de um produto como dispositivo médico com aprendizado de máquina, e que exigência recai sobre os dados de treinamento?",
     "O IMDRF é explícito quanto à ordem: um produto precisa primeiro atender à definição de dispositivo médico antes de poder ser um dispositivo habilitado por aprendizado de máquina. Quanto aos dados, as boas práticas exigem que os conjuntos de treinamento e de teste sejam selecionados e mantidos adequadamente independentes, tratando todas as fontes potenciais de dependência.",
     "hard"),

    ("good_ml_practice_guiding_principles_FDA_2021", "Health Canada, and the United Kingdom",
     "predetermined_change_control_FDA_2023", "monitored for performance and re-training risks",
     "Quem estabeleceu os princípios de boas práticas de aprendizado de máquina e o que eles esperam do dispositivo depois de implantado?",
     "Os 10 princípios orientadores de boas práticas foram identificados conjuntamente pelo FDA, pela Health Canada e pela MHRA do Reino Unido. Entre as expectativas regulatórias alinhadas a essas práticas está o monitoramento de desempenho dos modelos implantados e a gestão dos riscos de retreinamento.",
     "medium"),

    ("dpia_ICO_", "The risk to the rights and freedoms of natural persons",
     "de_identifying_government_datasets_NIST_2023", "Re-identifcation probability",
     "Como se mede o risco a que o titular fica exposto no tratamento e na divulgação de dados?",
     "O guia da ICO remete ao Considerando 75, que liga risco a dano potencial às pessoas: o risco aos direitos e liberdades das pessoas singulares, de probabilidade e gravidade variáveis, pode resultar de tratamento que cause dano físico, material ou imaterial. O NIST trabalha com uma medida específica para divulgação de dados: a probabilidade de reidentificação, isto é, a probabilidade de a identidade de um indivíduo ser corretamente inferida por um terceiro externo.",
     "hard"),

    # ── Constituição, códigos e a norma infralegal que os concretiza ────────
    ("cf_BR_1988", "A saúde é direito de todos e dever do Estado",
     "lei_8080_BR_90", "utilização da epidemiologia para o estabelecimento de prioridades",
     "Qual o fundamento constitucional do direito à saúde e como a lei do SUS traduz isso em critério de prioridade?",
     "A Constituição estabelece que a saúde é direito de todos e dever do Estado, garantido mediante políticas sociais e econômicas que reduzam o risco de doença e assegurem acesso universal e igualitário às ações e serviços. A Lei nº 8.080/1990 desce ao critério operacional: a epidemiologia é o instrumento para estabelecer prioridades, alocar recursos e orientar programaticamente.",
     "hard"),

    ("cf_BR_1988", "são invioláveis a intimidade, a vida privada",
     "lgpd_BR_2018", "anonimização: utilização de meios técnicos razoáveis e disponíveis",
     "Onde a proteção da vida privada tem assento constitucional e que técnica a LGPD oferece para afastar o dado do indivíduo?",
     "A Constituição declara invioláveis a intimidade, a vida privada, a honra e a imagem das pessoas, assegurando indenização pelo dano material ou moral decorrente da violação. A LGPD opera no plano técnico: define anonimização como o uso de meios técnicos razoáveis e disponíveis no momento do tratamento pelos quais o dado perde a possibilidade de associação ao indivíduo.",
     "hard"),

    ("cp_BR_1940", "Revelar alguém, sem justa causa, segredo",
     "codigo_etica_medica_CFM_2019", "Revelar fato de que tenha conhecimento em virtude do exercício",
     "Que consequência penal e que consequência ética decorrem de revelar segredo obtido no exercício da profissão?",
     "O Código Penal tipifica a violação de segredo profissional: revelar alguém, sem justa causa, segredo de que tem ciência em razão de função, ministério, ofício ou profissão, cuja revelação possa produzir dano a outrem, com pena de detenção. O Código de Ética Médica veda a mesma conduta no plano deontológico, ressalvando motivo justo, dever legal ou consentimento escrito do paciente.",
     "hard"),

    ("cdc_BR_1990", "terá acesso às informações existentes em cadastros, fichas, registros",
     "gdpr_regulation_EU_2016", "the existence of automated decision-making, including profiling",
     "Que direito de acesso a registros o consumidor tem no Brasil e o que o RGPD acrescenta sobre decisão automatizada?",
     "O Código de Defesa do Consumidor garante acesso às informações existentes em cadastros, fichas, registros e dados pessoais e de consumo arquivados sobre o consumidor, bem como às respectivas fontes. O RGPD vai além do acesso ao registro: exige informar a existência de decisão automatizada, incluindo definição de perfis, com informação significativa sobre a lógica envolvida e as consequências previstas.",
     "hard"),

    ("eca_BR_1990", "O direito ao respeito consiste na inviolabilidade da integridade física",
     "codigo_etica_medica_CFM_2019", "Revelar sigilo profissional relacionado a paciente criança ou adolescente",
     "Como o ECA define o direito ao respeito da criança e o que o Código de Ética Médica diz sobre o sigilo dela?",
     "O ECA define o direito ao respeito como a inviolabilidade da integridade física, psíquica e moral da criança e do adolescente, abrangendo a preservação da imagem, da identidade, da autonomia, dos valores e dos espaços pessoais. O Código de Ética Médica veda revelar sigilo profissional de paciente criança ou adolescente com capacidade de discernimento, inclusive aos pais ou representantes legais, salvo quando a não revelação puder causar dano ao paciente.",
     "hard"),

    # ── Telemedicina, publicidade e IA na medicina ─────────────────────────
    ("resolucao_2314_CFM_2022", "consulta presencial é o padrão ouro",
     "resolucao_2454_CFM_2026", "não substituem a autoridade e a decisão final humana",
     "Que posição a telemedicina e a inteligência artificial ocupam em relação ao ato médico presencial e à decisão do médico?",
     "A Resolução CFM nº 2.314/2022 estabelece que a consulta presencial é o padrão ouro de referência, sendo a telemedicina ato complementar. A Resolução CFM nº 2.454/2026 aplica a mesma lógica de subordinação à IA: os sistemas servem de apoio ao médico, mas não substituem a autoridade e a decisão final humana sobre o cuidado.",
     "hard"),

    ("resolucao_2454_CFM_2026", "permanece integralmente responsável pelos atos médicos",
     "projeto_lei_2338_BR_2023", "A supervisão humana não será exigida caso",
     "Quem responde pelo ato praticado com auxílio de IA na medicina e em que hipótese a supervisão humana pode ser dispensada?",
     "A Resolução CFM nº 2.454/2026 é categórica: no campo da responsabilidade ético-profissional, o médico permanece integralmente responsável pelos atos praticados mediante uso de modelos, sistemas e aplicações de IA, e o uso da tecnologia não o exime do Código de Ética Médica. O PL 2338/2023 admite exceção no plano regulatório geral: a supervisão humana pode ser dispensada quando comprovadamente impossível ou de esforço desproporcional, com outras obrigações recaindo sobre o agente.",
     "hard"),

    ("resolucao_2336_CFM_2023", "divulgar, quando não especialista, que trata de sistemas orgânicos",
     "codigo_etica_medica_CFM_2019", "Fazer referência a casos clínicos identificáveis",
     "Que limites o CFM impõe à divulgação da atividade médica e à exposição de casos clínicos?",
     "A Resolução CFM nº 2.336/2023 veda ao médico divulgar, quando não for especialista, que trata de sistemas orgânicos, órgãos ou doenças específicas, por induzir confusão com a divulgação de especialidades. O Código de Ética Médica proíbe, no capítulo do sigilo, fazer referência a casos clínicos identificáveis ou exibir pacientes e imagens que os tornem reconhecíveis.",
     "hard"),

    ("resolucao_2454_CFM_2026", "avaliação preliminar com a finalidade de definir seu grau de risco",
     "ai_risk_management_framework_NIST_2023", "AI system impact assessment approaches",
     "Que avaliação de risco se exige de uma instituição médica que usa IA e o que o NIST diz sobre avaliação de impacto?",
     "A Resolução CFM nº 2.454/2026 obriga instituições médicas, públicas ou privadas, que desenvolvam ou usem IA a realizar avaliação preliminar para definir o grau de risco do sistema. O NIST observa que abordagens de avaliação de impacto ajudam os atores de IA a entender impactos ou danos potenciais dentro de contextos específicos.",
     "hard"),

    # ── OMS, OCDE e as estratégias nacionais ───────────────────────────────
    ("ethics_governance_ai_health_WHO_2021", "humans should remain in full control of health-care systems",
     "resolucao_2454_CFM_2026", "A transparência no uso de IA será promovida por indicadores",
     "O que a OMS entende por autonomia humana em saúde e como o CFM operacionaliza a transparência no uso de IA?",
     "Para a OMS, autonomia no contexto da IA em saúde significa que os humanos devem permanecer em pleno controle dos sistemas de saúde e das decisões médicas. A Resolução CFM nº 2.454/2026 dá forma concreta a um dos requisitos disso: a transparência será promovida por indicadores científicos comprobatórios de acurácia, eficácia e segurança e por relatórios acessíveis em linguagem simples.",
     "hard"),

    ("regulatory_considerations_ai_health_WHO_2024", "documentation and transparency, risk management",
     "ai_framework_convention_council_europe_2024", "Each Party shall establish or designate one or more effective mechanisms",
     "Que áreas a OMS elege como temas regulatórios da IA em saúde e o que a convenção do Conselho da Europa exige de cada Parte?",
     "A OMS organiza as considerações regulatórias em áreas gerais: documentação e transparência, gestão de risco e abordagem de ciclo de vida, uso pretendido e validação analítica e clínica, qualidade dos dados, privacidade e proteção, e engajamento e colaboração. A convenção do Conselho da Europa fixa uma obrigação institucional: cada Parte deve estabelecer ou designar um ou mais mecanismos efetivos de supervisão do cumprimento.",
     "hard"),

    ("ai_principles_OECD_", "An AI system is a machine-based system",
     "projeto_lei_2338_BR_2023", "sistemas que, uma vez ativados, podem selecionar e atacar alvos",
     "Como a OCDE define sistema de inteligência artificial e que subcategoria a proposta brasileira destaca à parte?",
     "A OCDE define sistema de IA como sistema baseado em máquina que, para objetivos explícitos ou implícitos, infere a partir da entrada que recebe como gerar saídas — previsões, conteúdo, recomendações ou decisões — capazes de influenciar ambientes físicos ou virtuais. O PL 2338/2023 destaca, dentro do universo assim definido, os sistemas de armas autônomas: os que, uma vez ativados, podem selecionar e atacar alvos sem intervenção humana adicional.",
     "hard"),

    ("plano_brasileiro_ia_BR_2025", "inclusão social e oferecendo soluções tangíveis",
     "estrategia_brasileira_ia_MCTI_2021", "deve ter por objetivo potencializar o desenvolvimento",
     "Que objetivo o Plano Brasileiro de IA declara e como ele se relaciona com o da Estratégia Brasileira de IA?",
     "O Plano Brasileiro de Inteligência Artificial declara como objetivo garantir que a IA melhore a vida do povo brasileiro, promovendo inclusão social e oferecendo soluções tangíveis em áreas prioritárias como saúde e educação. A Estratégia Brasileira de IA, anterior, formula o objetivo em termos de potencializar o desenvolvimento e a utilização da tecnologia para promover o avanço científico e resolver problemas concretos do País.",
     "medium"),

    ("relatorio_pesquisa_CEP_", "Esse formulário se aplica para pesquisas nas áreas das ciências sociais",
     "regulamento_interno_CEP_2024", "Recebidos os relatórios parciais ou finais",
     "A que tipo de pesquisa se aplica o modelo de relatório do CEP e o que acontece quando ele é recebido?",
     "O modelo de relatório de pesquisa do CEP aplica-se a pesquisas nas áreas das ciências sociais e humanidades; para estudo clínico, o modelo sugerido é o da Conep. Recebidos os relatórios parciais ou finais, o regulamento do CEP-UFMG atribui ao coordenador ou a membro do colegiado a responsabilidade pela apreciação.",
     "medium"),

    # ── Os dois documentos que faltavam cobrir ─────────────────────────────
    ("aiml_samd_action_plan_FDA_2021", "they are vulnerable to bias",
     "ai_risk_management_framework_NIST_2023", "Decisions that go into the design, development, deployment",
     "Por que sistemas de IA treinados com dados históricos são vulneráveis a viés, segundo o FDA e o NIST?",
     "O plano de ação do FDA aponta a origem no dado: como os sistemas de IA/ML são desenvolvidos e treinados a partir de conjuntos de dados históricos, ficam vulneráveis a viés. O NIST acrescenta a origem humana: as decisões de projeto, desenvolvimento, implantação, avaliação e uso refletem vieses sistêmicos e cognitivos dos atores envolvidos.",
     "hard"),

    ("global_digital_health_strategy_WHO_2020_2027", "internationally connected digital health system",
     "rnds_MS_", "plataforma oficial de interoperabilidade",
     "Que sistema de saúde digital a OMS propõe construir e que realização concreta o Brasil tem nessa direção?",
     "A estratégia global da OMS propõe contribuir para a construção de um sistema de saúde digital conectado internacionalmente, considerando os riscos potenciais, e apoiar os países em seus programas nacionais. No Brasil, a Rede Nacional de Dados em Saúde materializa essa camada nacional: é a plataforma oficial de interoperabilidade do Ministério da Saúde para conectar sistemas de saúde em todo o país.",
     "hard"),

    ("global_digital_health_strategy_WHO_2020_2027", "digital health literacy, gender equality",
     "estrategia_brasileira_ia_MCTI_2021", "É fundamental que cada cidadão domine habilidades digitais",
     "Que lugar a literacia digital ocupa na estratégia da OMS e na estratégia brasileira de IA?",
     "A OMS põe a literacia digital num objetivo estratégico próprio, ao lado da igualdade de gênero, do empoderamento das mulheres e de abordagens inclusivas de adoção de tecnologias, colocando as pessoas no centro. A Estratégia Brasileira de IA converge: afirma ser fundamental que cada cidadão domine habilidades digitais básicas e tenha competências-chave aplicáveis a diversas atividades profissionais.",
     "hard"),

    # ── Cruzamentos entre blocos que ainda não se encontraram ──────────────
    ("lgpd_BR_2018", "encarregado: pessoa indicada pelo controlador para atuar como canal",
     "resolucao_2_ANPD_2024", "podem cumprir a obrigação de elaboração e manutenção de registro",
     "Que função a LGPD cria para intermediar controlador e titular, e que alívio a ANPD dá ao agente de pequeno porte?",
     "A LGPD define o encarregado como a pessoa indicada pelo controlador para atuar como canal de comunicação entre o controlador, os titulares dos dados e a autoridade nacional. Para o agente de tratamento de pequeno porte, a ANPD admite forma simplificada de cumprir a obrigação de elaborar e manter o registro das operações de tratamento.",
     "hard"),

    ("lei_14874_BR_2024", "biobanco: coleção organizada, sem fins comerciais",
     "de_identifying_government_datasets_NIST_2023", "if specifc identifers are not needed for maintenance",
     "Como a lei brasileira define biobanco e que princípio o NIST aplica à coleta de identificadores?",
     "A Lei nº 14.874/2024 define biobanco como coleção organizada, sem fins comerciais, de material biológico humano e informações associadas, coletados e armazenados para fins de pesquisa, sob responsabilidade e gerenciamento de instituição. O NIST formula o princípio correspondente no plano do dado: se identificadores específicos não são necessários para manutenção, síntese e uso, eles não devem ser coletados.",
     "hard"),

    ("cdc_BR_1990", "respondem, independentemente da existência de culpa",
     "resolucao_2454_CFM_2026", "permanece integralmente responsável pelos atos médicos",
     "Como se distribui a responsabilidade por defeito de produto no direito do consumidor e por ato médico apoiado em IA?",
     "O Código de Defesa do Consumidor adota responsabilidade independente de culpa pela reparação dos danos causados por defeitos decorrentes de projeto, fabricação, construção e montagem. A Resolução CFM nº 2.454/2026 mantém a responsabilidade no profissional: o médico permanece integralmente responsável pelos atos praticados mediante uso de modelos, sistemas e aplicações de IA.",
     "hard"),

    ("eca_BR_1990", "É dever de todos velar pela dignidade da criança",
     "multimodal_models_guidance_WHO_2024", "technology-facilitated gender-based violence",
     "Que dever geral o ECA impõe quanto à dignidade da criança e que risco a OMS associa aos modelos multimodais?",
     "O ECA estabelece que é dever de todos velar pela dignidade da criança e do adolescente, pondo-os a salvo de qualquer tratamento desumano, violento, aterrorizante, vexatório ou constrangedor. A OMS alerta que os modelos multimodais podem ampliar a violência de gênero facilitada por tecnologia, incluindo cyberbullying, discurso de ódio e uso não consentido de imagens e vídeos, com implicações sérias para adolescentes e mulheres.",
     "hard"),

    ("cp_BR_1940", "Revelar alguém, sem justa causa, segredo",
     "gdpr_regulation_EU_2016", "subject to a duty of professional secrecy both during",
     "Que dever de sigilo recai sobre quem exerce função pública de supervisão de dados e sobre o profissional em geral?",
     "O RGPD impõe aos membros e ao pessoal de cada autoridade de controlo dever de sigilo profissional durante e após o mandato, quanto a qualquer informação confidencial conhecida no exercício das funções. O Código Penal brasileiro dá à violação equivalente natureza criminal: revelar, sem justa causa, segredo de que se tem ciência em razão de função, ministério, ofício ou profissão, quando a revelação possa produzir dano.",
     "hard"),

    ("resolucao_1627_CFM_2001", "A responsabilidade civil é exercida sempre que há dano evitável",
     "cdc_BR_1990", "respondem, independentemente da existência de culpa",
     "Que pressuposto a responsabilidade civil do profissional exige e em que ela difere da responsabilidade por defeito de produto?",
     "A Resolução CFM nº 1.627/2001 condiciona a responsabilidade civil do profissional a dano evitável e a relação causal entre esse dano e um elemento de erro profissional — imperícia, imprudência ou negligência. O Código de Defesa do Consumidor dispensa esse exame: a reparação dos danos causados por defeitos de projeto, fabricação, construção ou montagem independe da existência de culpa.",
     "hard"),

    ("imdrf_good_ml_practice_2025", "Selected reference standards are fit-for-purpose",
     "imdrf_samd_clinical_evaluation_2017", "performance metrics for a SaMD have a scientific level of rigor",
     "Que exigência recai sobre os padrões de referência e sobre as métricas de desempenho de um software como dispositivo médico?",
     "As boas práticas do IMDRF exigem que os padrões de referência selecionados sejam adequados ao propósito, com métodos aceitos que assegurem dados clinicamente relevantes e bem caracterizados. Na avaliação clínica, os reguladores globais esperam que as métricas de desempenho do SaMD tenham rigor científico proporcional ao risco e ao impacto do dispositivo.",
     "hard"),

    ("justice_data_governance_OECD_2024", "A justice system that has as its purpose and design",
     "cf_BR_1988", "A saúde é direito de todos e dever do Estado",
     "Como a OCDE define justiça centrada nas pessoas e que formulação equivalente a Constituição usa para a saúde?",
     "A OCDE define justiça centrada nas pessoas como o sistema cujo propósito e desenho visam atender igualmente às necessidades de todas as pessoas, habilitando sua participação e engajamento efetivos. A Constituição brasileira usa formulação análoga para a saúde: direito de todos e dever do Estado, com acesso universal e igualitário às ações e serviços.",
     "hard"),

    ("ethical_impact_assessment_UNESCO_2023", "Remediability here refers to the capacity for reparability",
     "resolucao_2306_CFM_2022", "A suspeição poderá ser alegada, no prazo de 15",
     "Que noção de reparação a UNESCO aplica a impactos de IA e que instrumento processual o CFM oferece contra julgador parcial?",
     "A UNESCO define remediabilidade como a capacidade de reparação ou restauração — se e com que facilidade pessoas ou objetos impactados podem voltar a situação equivalente à anterior ao impacto. No processo ético-profissional, a correção se dá por instrumento próprio: a suspeição pode ser alegada em 15 dias contados do conhecimento do fato, em petição específica.",
     "medium"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    temas: dict[str, str] = {}
    for linha in (AQUI.parents[1] / "data/processed/documents.jsonl").open(encoding="utf-8"):
        r = json.loads(linha)
        temas.setdefault(r["metadata"]["document_id"], r["metadata"].get("theme"))

    registros, falhas = [], []
    for i, (da, aa, db, ab, pergunta, resposta, dif) in enumerate(ESPEC, start=1):
        try:
            ev = [citar(da, aa), citar(db, ab)]
        except AncoraRuim as erro:
            falhas.append((i, str(erro)))
            continue
        registros.append({
            "n": f"c{len(registros)+1:03d}",
            "question": pergunta,
            "answer": resposta,
            "evidencia": ev,
            "n_evidencias": 2,
            "documento": None,
            "documentos": [da, db],
            "forma": None,
            "janela": None,
            "passagem_texto": None,
            "question_type": "comparative",
            "difficulty": dif,
            "theme": f"{temas.get(da, '')} | {temas.get(db, '')}",
            "source_lang": "mixed",
            "origem": "comparativas_cruzadas",
        })

    print(f"comparativas montadas: {len(registros)} de {len(ESPEC)}")
    if falhas:
        print(f"\nâncoras a corrigir: {len(falhas)}")
        for i, e in falhas:
            print(f"   [{i}] {e}")
    docs = Counter(d for r in registros for d in r["documentos"])
    print(f"\ndocumentos citados: {len(docs)}")
    print(f"  {dict(Counter(r['difficulty'] for r in registros))}")

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
