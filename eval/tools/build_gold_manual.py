#!/usr/bin/env python3
"""
build_gold_manual.py
════════════════════
Monta o GOLD SET autoral: perguntas escritas por uma LLM independente (Claude,
em sessão) a partir de trechos REAIS do corpus. Os qrels são validados contra o
documents.jsonl — se qualquer chunk_id não existir, o script ABORTA e lista os
inválidos (evita o descasamento que quebrou o gabarito antigo).

As perguntas ficam na lista QUESTIONS abaixo (crescendo em lotes por tipo).
Saída: data/golden_qa.jsonl (+ queries.csv, qrels.csv), mesmo formato do 01.

Uso:  python eval/tools/build_gold_manual.py
"""
from __future__ import annotations
import sys, json, csv
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.common import (load_config, load_corpus, resolve, ensure_dir, setup_io,
                        neighbors_same_doc)
from lib.retrievers import detect_lang

REF_UNANS = ("Não encontrei informações suficientes nas fontes recuperadas "
             "para responder com segurança.")

def F(q, a, qrels, diff="medium"):
    return dict(question=q, reference_answer=a, question_type="factual",
                difficulty=diff, qrels=qrels)

def U(q):
    return dict(question=q, reference_answer=REF_UNANS, question_type="unanswerable",
                difficulty="medium", qrels={})

def MH(q, a, qrels, diff="medium"):
    return dict(question=q, reference_answer=a, question_type="multi_hop",
                difficulty=diff, qrels=qrels)

def C(q, a, qrels, diff="hard"):
    return dict(question=q, reference_answer=a, question_type="comparative",
                difficulty=diff, qrels=qrels)

QUESTIONS = [
 # ═══════════ FACTUAL (25) — respondível por 1 trecho ═══════════
 F("No acompanhamento de doenças crônicas por teleconsulta, é obrigatório algum atendimento presencial?",
   "Sim. A Resolução CFM nº 2.314/2022 exige, no acompanhamento de doenças crônicas ou que demandem acompanhamento por longo tempo, consulta presencial com o médico assistente em intervalos não superiores a 180 dias.",
   {"resolucao_2314_CFM_2022_p4_c2": 2}, "easy"),
 F("No atendimento por telemedicina, o que precisa ser garantido quanto ao consentimento do paciente sobre o compartilhamento de suas informações?",
   "Deve ser assegurado consentimento explícito, com o paciente ou seu representante legal ciente de que suas informações pessoais podem ser compartilhadas e de que tem o direito de negar essa permissão, salvo em situação de emergência médica.",
   {"resolucao_2314_CFM_2022_p7_c0": 2}, "easy"),
 F("Na telemedicina, o compartilhamento das informações do paciente com outro profissional exige o quê?",
   "Prévia permissão do paciente, mediante consentimento livre e esclarecido, e protocolos de segurança capazes de garantir a confidencialidade e a integridade das informações.",
   {"resolucao_2314_CFM_2022_p2_c2": 2}),
 F("Preciso do consentimento do titular para tratar dados pessoais que ele próprio tornou manifestamente públicos?",
   "Não. A LGPD dispensa a exigência de consentimento para o tratamento de dados tornados manifestamente públicos pelo próprio titular, resguardados os seus direitos e os princípios previstos na Lei.",
   {"lgpd_BR_2018_c18": 2}),
 F("Quais direitos tem uma pessoa afetada por um sistema de IA, segundo o PL 2338/2023, independentemente do grau de risco do sistema?",
   "Direito à informação sobre suas interações com sistemas de IA (de forma acessível, gratuita e compreensível, inclusive quanto ao caráter automatizado); direito à privacidade e à proteção de dados pessoais nos termos da LGPD; e direito à não discriminação ilícita ou abusiva e à correção de vieses discriminatórios.",
   {"projeto_lei_2338_BR_2023_p6_c1": 2}),
 F("No PL 2338/2023, qual autoridade coordenaria o Sistema Nacional de Regulação e Governança de Inteligência Artificial (SIA)?",
   "A Autoridade Nacional de Proteção de Dados (ANPD), indicada como autoridade competente que coordenará o SIA.",
   {"projeto_lei_2338_BR_2023_p23_c0": 2}),
 F("O PL 2338/2023 prevê tratamento diferenciado para startups e pequenas empresas?",
   "Sim. As autoridades setoriais deverão definir critérios diferenciados para sistemas de IA ofertados por microempresas, empresas de pequeno porte e startups, considerando o impacto concorrencial, o número de usuários afetados e a natureza das atividades econômicas.",
   {"projeto_lei_2338_BR_2023_p32_c1": 2}),
 F("Um médico que atua como auditor pode receber remuneração vinculada ao sucesso da causa ou à glosa?",
   "Não. O Código de Ética Médica veda ao médico, na função de auditor ou perito, receber remuneração ou gratificação vinculada à glosa ou ao sucesso da causa, e exige que atue com absoluta isenção.",
   {"codigo_etica_medica_CFM_2019_p41_c1": 2}),
 F("Como o Código de Ética Médica trata a responsabilidade do médico por seus atos profissionais?",
   "O médico se responsabiliza, em caráter pessoal e nunca presumido, pelos seus atos profissionais, resultantes de relação de confiança e executados com diligência, competência e prudência.",
   {"codigo_etica_medica_CFM_2019_p19_c0": 2}),
 F("A partir de quando passou a vigorar o novo Código de Ética Médica (Resolução CFM nº 2.217/2018)?",
   "A partir de 30 de abril de 2019, atualizando a versão anterior, de 2009.",
   {"codigo_etica_medica_CFM_2019_p9_c0": 2}, "easy"),
 F("As cláusulas mandatórias para transferência internacional de dados aprovadas pela ANPD podem ser alteradas pelas partes?",
   "Não. As cláusulas mandatórias devem ser adotadas integralmente e sem qualquer alteração em seu texto, a fim de assegurar a validade da transferência internacional de dados.",
   {"resolucao_19_ANPD_2024_c44": 2}),
 F("Um sistema de IA transparente é necessariamente preciso, seguro e justo?",
   "Não. Segundo o NIST AI RMF, um sistema transparente não é necessariamente preciso, com privacidade, seguro ou justo; porém, em sistemas opacos é difícil determinar se essas características estão presentes e como evoluem ao longo do tempo.",
   {"ai_risk_management_framework_NIST_2023_p21_c0": 2}),
 F("O que o NIST AI RMF reconhece sobre os riscos das tecnologias de IA?",
   "Que, além do potencial de transformar positivamente a sociedade, as tecnologias de IA também apresentam riscos que podem impactar negativamente indivíduos, grupos, organizações, comunidades, a sociedade e o meio ambiente.",
   {"ai_risk_management_framework_NIST_2023_p6_c0": 2}, "easy"),
 F("Há usos de IA que a Recomendação da UNESCO sobre ética da IA proíbe?",
   "Sim. A Recomendação proíbe o uso de IA para pontuação social (social scoring) e para vigilância em massa; e, em decisões com impacto irreversível ou que envolvam vida e morte, exige que a determinação humana final se aplique.",
   {"ethical_impact_assessment_UNESCO_2023_p13_c0": 2}),
 F("Como a ferramenta de avaliação de impacto ético da UNESCO classifica a gravidade de um impacto negativo?",
   "Em quatro níveis, num contínuo: Moderado/Menor, Sério, Crítico e Catastrófico. O nível Catastrófico abrange, por exemplo, a privação do direito à vida e danos irreversíveis à integridade física, psicológica ou moral.",
   {"ethical_impact_assessment_UNESCO_2023_p45_c0": 2}),
 F("O direito à portabilidade de dados do GDPR se aplica quando o tratamento se baseia em obrigação legal do controlador?",
   "Não. O direito à portabilidade aplica-se quando o tratamento se baseia no consentimento ou em contrato; não se aplica quando fundado em outra base, como o cumprimento de obrigação legal ou o exercício de função de interesse público.",
   {"gdpr_regulation_EU_2016_p13_c3": 2}),
 F("Que capacidade de segurança o GDPR exige diante de um incidente físico ou técnico?",
   "A capacidade de restaurar, em tempo hábil, a disponibilidade e o acesso aos dados pessoais em caso de incidente físico ou técnico, além de assegurar a confidencialidade, integridade, disponibilidade e resiliência dos sistemas de tratamento.",
   {"gdpr_regulation_EU_2016_p52_c0": 2}),
 F("A Estratégia Brasileira de Inteligência Artificial menciona alguma ação relacionada ao viés algorítmico?",
   "Sim. A EBIA prevê desenvolver técnicas para identificar e tratar o risco de viés algorítmico, além de estimular a transparência e a observância, pelos sistemas de IA, de direitos humanos, valores democráticos e da diversidade.",
   {"estrategia_brasileira_ia_MCTI_2021_p24_c1": 2}),
 F("O que a OMS entende por 'responsividade' (responsiveness) em inteligência artificial para a saúde?",
   "Exige que projetistas, desenvolvedores e usuários examinem contínua, sistemática e transparentemente a tecnologia de IA para verificar se ela responde de forma adequada e apropriada às expectativas e aos requisitos comunicados no contexto em que é usada.",
   {"ethics_governance_ai_health_WHO_2021_p46_c1": 2}),
 F("Que exemplos de controle a FDA sugere contra ataques de envenenamento de dados (data poisoning) em dispositivos habilitados por IA?",
   "Validar, autenticar e limpar os dados; empregar detecção de anomalias e verificações de integridade (por exemplo, hashes criptográficos); e aplicar treinamento adversarial para melhorar a robustez e a segurança dos modelos.",
   {"ai_device_software_guidance_FDA_2025_p39_c0": 2}),
 F("Que informações sobre a população de pacientes a FDA sugere descrever para um dispositivo habilitado por IA?",
   "A demografia da população de pacientes (por exemplo, sexo, idade, raça, etnia, altura e peso) e as condições e equipamentos de aquisição de dados, incluindo fatores que possam impactar os sinais analisados.",
   {"ai_device_software_guidance_FDA_2025_p25_c2": 2}),
 F("Quais são responsabilidades do Comitê de Ética em Pesquisa (CEP) segundo a Lei nº 14.874/2024?",
   "Assegurar os direitos, a segurança e o bem-estar dos participantes (especialmente os vulneráveis), considerar a qualificação do pesquisador para a pesquisa proposta e conduzir a análise da pesquisa submetida e o monitoramento de sua execução.",
   {"lei_14874_BR_2024_c27": 2}),
 F("Qual é um dos objetivos do Espaço Europeu de Dados de Saúde (EEDS)?",
   "Melhorar o funcionamento do mercado interno por meio de um regime jurídico e técnico uniforme para o desenvolvimento, a comercialização e o uso de sistemas de registos de saúde eletrónicos, além de melhorar o acesso e o controlo das pessoas sobre os seus dados de saúde eletrónicos.",
   {"european_health_data_space_EU_2025_p1_c2": 2}),
 F("No NIST AI RMF, o que a subcategoria MAP 1.6 recomenda sobre os requisitos do sistema?",
   "Que os requisitos do sistema (por exemplo, 'o sistema deve respeitar a privacidade dos usuários') sejam levantados e compreendidos pelos atores de IA relevantes, com as decisões de projeto considerando implicações sociotécnicas para tratar os riscos de IA.",
   {"ai_risk_management_framework_NIST_2023_p31_c1": 2}, "hard"),
 F("Segundo o PL 2338/2023, os agentes de IA de alto risco têm algum dever quanto às medidas de governança?",
   "Sim. Devem garantir que seus sistemas estejam de acordo com as medidas de governança definidas no Capítulo IV da Lei, bem como com outras legislações pertinentes, em especial as do respectivo setor.",
   {"projeto_lei_2338_BR_2023_p14_c3": 2}),

 # ═══════════ UNANSWERABLE (25) — fora do escopo do acervo ═══════════
 U("Qual é a multa máxima prevista pelo AI Act da União Europeia para o uso de práticas de IA proibidas?"),
 U("Quantos processos por erro médico envolvendo inteligência artificial foram julgados no Brasil em 2024?"),
 U("Qual é o preço de mercado de uma solução de IA para diagnóstico por imagem no Brasil?"),
 U("A lei de saúde dos Estados Unidos (HIPAA) permite usar dados de pacientes para treinar modelos de IA sem autorização?"),
 U("Um paciente específico pode processar o hospital por um erro de diagnóstico cometido pelo algoritmo no seu caso?"),
 U("Qual foi a acurácia do modelo de IA aprovado pela Anvisa para triagem de retinopatia diabética em 2023?"),
 U("Quantos hospitais no Brasil já utilizam prontuário eletrônico com inteligência artificial em 2025?"),
 U("O que a lei de proteção de dados da China (PIPL) exige para a transferência internacional de dados de saúde?"),
 U("Quais são os requisitos da Anvisa para o registro de um software como dispositivo médico com IA?"),
 U("Quanto custa obter a certificação ISO/IEC 42001 para um sistema de gestão de inteligência artificial?"),
 U("Qual universidade brasileira oferece o melhor curso de inteligência artificial aplicada à saúde?"),
 U("O NHS do Reino Unido exige avaliação de impacto algorítmico para sistemas de IA na saúde pública?"),
 U("Qual é a alíquota do ISS sobre serviços de inteligência artificial em saúde no município de São Paulo?"),
 U("Como devo configurar o banco vetorial Qdrant para indexar os embeddings do modelo BGE-m3?"),
 U("Qual é a pena para exercício ilegal da medicina prevista no Código Penal argentino?"),
 U("O FDA já aprovou algum modelo de linguagem de grande porte (LLM) para diagnóstico autônomo até 2025?"),
 U("Qual é o salário médio de um auditor de inteligência artificial responsável no Brasil?"),
 U("O Japão possui uma lei específica de responsabilidade civil por danos causados por IA médica?"),
 U("Quantas reclamações sobre vazamento de dados de saúde a ANPD recebeu em 2024?"),
 U("Qual é a taxa de falso positivo máxima aceitável para reconhecimento facial segundo a legislação federal brasileira?"),
 U("Para um chatbot de RAG jurídico, qual banco de dados vetorial é melhor: Qdrant ou Pinecone?"),
 U("Qual é a definição de 'sistema de IA de risco excessivo' no regulamento de inteligência artificial de Singapura?"),
 U("Quantos médicos foram punidos pelo Conselho Federal de Medicina por uso indevido de IA em 2023?"),
 U("Que dose de insulina devo prescrever para um paciente diabético específico com base nos dados dele?"),
 U("Qual é o percentual de leitos hospitalares com IA preditiva no SUS em 2025?"),

 # ═══════════ MULTI-HOP (25) — exige combinar 2 trechos do mesmo documento ═══════════
 MH("Uma pessoa afetada por um sistema de IA de alto risco: que direitos ela tem e a que medidas de governança esse sistema está sujeito, segundo o PL 2338/2023?",
    "Independentemente do grau de risco, tem direito à informação sobre as interações com a IA, à privacidade e proteção de dados (LGPD) e à não discriminação e correção de vieses (Art. 5º). Por ser de alto risco, o sistema deve ainda estar em conformidade com as medidas de governança do Capítulo IV da Lei e das legislações setoriais pertinentes (Art. 21).",
    {"projeto_lei_2338_BR_2023_p6_c1": 2, "projeto_lei_2338_BR_2023_p14_c3": 2}),
 MH("Na telemedicina, o que a Resolução CFM 2.314/2022 exige quanto ao consentimento para compartilhar informações e quanto ao acompanhamento presencial de doenças crônicas?",
    "Exige consentimento explícito, com o paciente ciente de que suas informações podem ser compartilhadas e do direito de negar (salvo emergência); e, para doenças crônicas ou de longo acompanhamento, consulta presencial com o médico assistente em intervalos não superiores a 180 dias.",
    {"resolucao_2314_CFM_2022_p7_c0": 2, "resolucao_2314_CFM_2022_p4_c2": 2}),
 MH("Que deveres o Código de Ética Médica impõe ao médico quanto à responsabilidade por seus atos e quanto à atuação como auditor?",
    "O médico responde pessoalmente, e nunca de forma presumida, por seus atos, executados com diligência, competência e prudência; e, na função de auditor, deve atuar com absoluta isenção, sendo-lhe vedado receber remuneração vinculada à glosa ou ao sucesso da causa.",
    {"codigo_etica_medica_CFM_2019_p19_c0": 2, "codigo_etica_medica_CFM_2019_p41_c1": 2}),
 MH("Por que medir riscos de IA é desafiador e por que esses riscos diferem dos de software tradicional, segundo o NIST?",
    "Falta consenso sobre métricas robustas e verificáveis, e as medições podem ser simplificadas demais, manipuladas ou ignorar diferenças entre grupos e contextos. Além disso, os riscos de IA são novos ou ampliados porque, por exemplo, os dados usados podem não representar adequadamente o contexto de uso — algo que os frameworks tradicionais não cobrem plenamente.",
    {"ai_risk_management_framework_NIST_2023_p11_c0": 2, "ai_risk_management_framework_NIST_2023_p43_c0": 2}),
 MH("Segundo a UNESCO, que usos de IA são proibidos e que mecanismos de responsabilização os sistemas de IA devem ter?",
    "São proibidos a pontuação social e a vigilância em massa, e decisões irreversíveis ou de vida e morte exigem determinação humana final. Os sistemas devem ainda contar com mecanismos de supervisão, avaliação de impacto, auditoria e devida diligência — incluindo proteção a denunciantes — para assegurar transparência e responsabilização.",
    {"ethical_impact_assessment_UNESCO_2023_p13_c0": 2, "ethical_impact_assessment_UNESCO_2023_p36_c0": 2}),
 MH("No GDPR, quando se aplica o direito à portabilidade e que medidas de segurança o controlador deve garantir para os dados?",
    "A portabilidade aplica-se quando o tratamento se baseia em consentimento ou contrato (não em obrigação legal ou interesse público). Quanto à segurança, o controlador deve assegurar confidencialidade, integridade, disponibilidade e resiliência dos sistemas e a capacidade de restaurar o acesso aos dados em tempo hábil após incidentes.",
    {"gdpr_regulation_EU_2016_p13_c3": 2, "gdpr_regulation_EU_2016_p52_c0": 2}),
 MH("A Estratégia Brasileira de IA trata tanto do viés algorítmico quanto do impacto no trabalho — o que propõe em cada frente?",
    "Para o viés, prevê desenvolver técnicas para identificar e tratar o risco de viés algorítmico e estimular transparência e respeito a direitos humanos. Para o trabalho, reconhece que novos e muitos empregos exigirão novas habilidades e enfatiza a capacitação contínua para preparar e readequar a força de trabalho.",
    {"estrategia_brasileira_ia_MCTI_2021_p24_c1": 2, "estrategia_brasileira_ia_MCTI_2021_p35_c1": 2}),
 MH("Para um dispositivo habilitado por IA, que informações sobre os pacientes descrever e que controles de cibersegurança considerar contra envenenamento de dados, segundo a FDA?",
    "Descrever a demografia dos pacientes (sexo, idade, raça, etnia, altura, peso) e as condições e equipamentos de aquisição de dados. Contra o envenenamento de dados, validar, autenticar e limpar os dados, usar detecção de anomalias e verificações de integridade (hashes) e aplicar treinamento adversarial.",
    {"ai_device_software_guidance_FDA_2025_p25_c2": 2, "ai_device_software_guidance_FDA_2025_p39_c0": 2}),
 MH("Na pesquisa clínica, quais são as responsabilidades do Comitê de Ética em Pesquisa (CEP) e as do pesquisador, segundo a Lei nº 14.874/2024?",
    "O CEP deve assegurar direitos, segurança e bem-estar dos participantes (sobretudo vulneráveis), avaliar a qualificação do pesquisador e conduzir a análise e o monitoramento da pesquisa. O pesquisador deve comprovar qualificação e experiência, obedecer às boas práticas clínicas e às exigências regulatórias e submeter a documentação (e emendas) à aprovação do CEP.",
    {"lei_14874_BR_2024_c27": 2, "lei_14874_BR_2024_c45": 2}),
 MH("Na transferência internacional de dados, o que a Resolução ANPD nº 19/2024 estabelece sobre as cláusulas mandatórias e sobre a responsabilização por danos ao titular?",
    "As cláusulas mandatórias devem ser adotadas integralmente e sem alteração, para garantir a validade da transferência. Quanto a danos, o juiz pode inverter o ônus da prova em favor do titular quando a alegação for verossímil ou houver hipossuficiência, e a parte que reparar o dano tem direito de regresso contra os demais responsáveis.",
    {"resolucao_19_ANPD_2024_c44": 2, "resolucao_19_ANPD_2024_c62": 2}),
 MH("Que sistema nacional o PL 2338/2023 autoriza para a governança da IA e que tratamento diferenciado prevê para startups?",
    "Autoriza o Sistema Nacional de Regulação e Governança de IA (SIA), coordenado pela ANPD e integrado por autoridades setoriais e conselhos específicos. Para startups, microempresas e empresas de pequeno porte, as autoridades setoriais devem definir critérios diferenciados, considerando impacto concorrencial, número de usuários afetados e a natureza das atividades.",
    {"projeto_lei_2338_BR_2023_p23_c0": 2, "projeto_lei_2338_BR_2023_p32_c1": 2}),
 MH("Segundo a OMS, que risco surge quando grandes empresas de tecnologia controlam dados e algoritmos de saúde, e como as parcerias público-privadas (PPPs) se inserem nesse cenário?",
    "Há um desequilíbrio de poder: essas empresas podem passar a governar decisões que caberiam a indivíduos, sociedades e governos, dado seu controle sobre dados e infraestrutura. Nas PPPs, o setor público costuma fornecer dados (como prontuários eletrônicos) que as empresas usam para desenvolver produtos, cabendo aos governos supervisionar esses arranjos.",
    {"ethics_governance_ai_health_WHO_2021_p79_c3": 2, "ethics_governance_ai_health_WHO_2021_p114_c1": 2}),
 MH("Por que a Resolução CFM 2.314/2022 considera o atendimento presencial a referência e o que exige para compartilhar informações do paciente na telemedicina?",
    "Seguindo recomendação da Associação Médica Mundial, considera o atendimento presencial o padrão ouro de referência, devendo as tecnologias servir para melhorar (e não substituir) a medicina. Para compartilhar informações com outro profissional, exige prévia permissão do paciente, consentimento livre e esclarecido e protocolos de segurança que garantam confidencialidade e integridade.",
    {"resolucao_2314_CFM_2022_p9_c0": 2, "resolucao_2314_CFM_2022_p2_c2": 2}),
 MH("Segundo o NIST AI RMF, qual a relação entre transparência e responsabilização (accountability) e como a função MAP trata os requisitos do sistema?",
    "Transparência não garante precisão, segurança ou justiça, mas ajuda a avaliar e a responsabilizar; quando as consequências são graves, desenvolvedores e implantadores devem reforçar proporcionalmente transparência e responsabilização. Na função MAP, requisitos como 'o sistema deve respeitar a privacidade dos usuários' devem ser levantados e compreendidos pelos atores relevantes, considerando implicações sociotécnicas.",
    {"ai_risk_management_framework_NIST_2023_p21_c0": 2, "ai_risk_management_framework_NIST_2023_p31_c1": 2}),
 MH("A Recomendação da UNESCO proíbe certos usos de IA e classifica a gravidade dos impactos — o que proíbe e quais são os níveis de gravidade?",
    "Proíbe o uso de IA para pontuação social e vigilância em massa, e exige determinação humana final em decisões irreversíveis ou de vida e morte. A gravidade dos impactos negativos é classificada em quatro níveis: Moderado/Menor, Sério, Crítico e Catastrófico.",
    {"ethical_impact_assessment_UNESCO_2023_p13_c0": 2, "ethical_impact_assessment_UNESCO_2023_p45_c0": 2}),
 MH("Na pesquisa clínica, o que é a instância nacional de ética em pesquisa e quais são as responsabilidades do CEP, segundo a Lei nº 14.874/2024?",
    "A instância nacional de ética em pesquisa é um colegiado interdisciplinar e independente, integrante do Ministério da Saúde, de caráter normativo, consultivo, deliberativo e educativo, que regula, fiscaliza e controla eticamente a pesquisa. O CEP deve assegurar direitos, segurança e bem-estar dos participantes, avaliar a qualificação do pesquisador e conduzir a análise e o monitoramento da pesquisa.",
    {"lei_14874_BR_2024_c10": 2, "lei_14874_BR_2024_c27": 2}),
 MH("Que responsabilidades cabem ao pesquisador e o que a Lei nº 14.874/2024 veda quanto ao material biológico armazenado em biobanco?",
    "O pesquisador deve ter e comprovar qualificação e experiência, obedecer às boas práticas clínicas e às exigências regulatórias e submeter a documentação ao CEP. Quanto ao material biológico humano em biobanco, é vedada a sua compra, venda ou qualquer comercialização — não se considerando comercialização a mera recuperação de custos de insumos, processamento, armazenamento e transporte.",
    {"lei_14874_BR_2024_c45": 2, "lei_14874_BR_2024_c63": 2}),
 MH("Segundo o NIST, os sistemas de IA só trazem benefícios, e é fácil medir seus riscos?",
    "Não. Além de benefícios, a IA apresenta riscos que podem impactar negativamente indivíduos, grupos, organizações, a sociedade e o meio ambiente. E medir esses riscos é desafiador: falta consenso sobre métricas robustas, que podem ser simplificadas demais, manipuladas ou deixar de considerar diferenças entre grupos e contextos.",
    {"ai_risk_management_framework_NIST_2023_p6_c0": 2, "ai_risk_management_framework_NIST_2023_p11_c0": 2}),
 MH("No GDPR, que características as sanções administrativas devem ter e que medidas de segurança dos dados são exigidas?",
    "As sanções devem ser efetivas, proporcionais e dissuasivas (podendo os Estados-Membros definir sua natureza, penal ou administrativa). Quanto à segurança, exige-se assegurar confidencialidade, integridade, disponibilidade e resiliência dos sistemas e a capacidade de restaurar o acesso aos dados em tempo hábil após incidentes.",
    {"gdpr_regulation_EU_2016_p28_c2": 2, "gdpr_regulation_EU_2016_p52_c0": 2}),
 MH("A Estratégia Brasileira de IA aborda tanto o viés algorítmico quanto o uso de IA na segurança pública — o que diz sobre cada um?",
    "Sobre o viés, prevê desenvolver técnicas para identificá-lo e tratá-lo, além de estimular transparência e respeito a direitos humanos. Sobre segurança pública, descreve aplicações analíticas de vídeo e áudio (por exemplo, detecção de armas, tiros ou tumultos) que podem auxiliar as autoridades a identificar ocorrências e priorizar urgências em tempo real.",
    {"estrategia_brasileira_ia_MCTI_2021_p24_c1": 2, "estrategia_brasileira_ia_MCTI_2021_p48_c1": 2}),
 MH("Para um dispositivo habilitado por IA, que obrigações de sistema de qualidade e que descrição da população de pacientes a FDA espera?",
    "Espera procedimentos para controlar produtos não conformes e implementar ações corretivas e preventivas, além de revisar periodicamente a adequação e a eficácia do sistema de qualidade. E espera a descrição da demografia dos pacientes (sexo, idade, raça, etnia, altura, peso) e das condições e equipamentos de aquisição de dados.",
    {"ai_device_software_guidance_FDA_2025_p11_c0": 2, "ai_device_software_guidance_FDA_2025_p25_c2": 2}),
 MH("A Resolução ANPD nº 19/2024: em que situações admite dados provenientes do exterior e que regra vale para as cláusulas mandatórias de transferência?",
    "Admite dados do exterior em casos como o mero trânsito de dados (sem comunicação ou uso compartilhado com agente no Brasil) ou o retorno ao país de proveniência, desde que este ofereça grau adequado de proteção reconhecido pela ANPD. Já as cláusulas mandatórias devem ser adotadas integralmente e sem qualquer alteração, para assegurar a validade da transferência.",
    {"resolucao_19_ANPD_2024_c10": 2, "resolucao_19_ANPD_2024_c44": 2}),
 MH("Segundo a OMS, o que exige a 'responsividade' de uma IA para saúde e que risco de poder existe quando poucas empresas controlam os dados de saúde?",
    "A responsividade exige examinar contínua, sistemática e transparentemente se a IA responde de forma adequada às necessidades e expectativas do contexto de uso, com dever de correção quando falha. Ao mesmo tempo, quando grandes empresas controlam a maior parte dos dados e algoritmos, cresce o risco de que passem a governar decisões que deveriam caber a indivíduos, sociedades e governos.",
    {"ethics_governance_ai_health_WHO_2021_p46_c1": 2, "ethics_governance_ai_health_WHO_2021_p79_c3": 2}),
 MH("No PL 2338/2023, além dos direitos das pessoas afetadas pela IA, que estrutura pública é criada para regular e governar a IA?",
    "As pessoas afetadas têm direito à informação, à privacidade e proteção de dados e à não discriminação. Para regular e governar a IA, o projeto autoriza o Sistema Nacional de Regulação e Governança de IA (SIA), coordenado pela ANPD e integrado por autoridades setoriais e conselhos específicos.",
    {"projeto_lei_2338_BR_2023_p6_c1": 2, "projeto_lei_2338_BR_2023_p23_c0": 2}),
 MH("A UNESCO exige mecanismos de responsabilização para a IA e classifica a gravidade dos impactos — o que exige e como classifica?",
    "Exige que os sistemas de IA sejam transparentes, explicáveis e responsáveis, com supervisão, avaliação de impacto, auditoria e devida diligência (incluindo proteção a denunciantes). A gravidade dos impactos negativos é avaliada em quatro níveis: Moderado/Menor, Sério, Crítico e Catastrófico.",
    {"ethical_impact_assessment_UNESCO_2023_p36_c0": 2, "ethical_impact_assessment_UNESCO_2023_p45_c0": 2}),

 # ═══════════ COMPARATIVE (25) — relaciona 2 documentos diferentes ═══════════
 C("Como o NIST AI RMF e a Recomendação da UNESCO tratam transparência e responsabilização (accountability) em IA?",
   "Para o NIST, a transparência não garante que um sistema seja preciso, seguro ou justo, mas ajuda a avaliá-lo e a responsabilizar os atores — que devem reforçar essas práticas quando as consequências são graves. A UNESCO exige que os sistemas sejam transparentes, explicáveis e responsáveis, com supervisão, avaliação de impacto, auditoria e devida diligência. Ambos ligam transparência a mecanismos de responsabilização.",
   {"ai_risk_management_framework_NIST_2023_p21_c0": 2, "ethical_impact_assessment_UNESCO_2023_p36_c0": 2}),
 C("Como o PL 2338/2023 e o GDPR abordam os direitos das pessoas sobre seus dados pessoais?",
   "O PL 2338 assegura à pessoa afetada por IA o direito à privacidade e à proteção de dados nos termos da LGPD, além de informação e não discriminação. O GDPR detalha direitos como a portabilidade, aplicável quando o tratamento se baseia em consentimento ou contrato. Ambos colocam a proteção de dados do titular como direito central, com o PL remetendo à LGPD.",
   {"projeto_lei_2338_BR_2023_p6_c1": 2, "gdpr_regulation_EU_2016_p13_c3": 2}),
 C("Como a Resolução CFM de telemedicina e a LGPD tratam a necessidade de consentimento para o uso de informações pessoais?",
   "Na telemedicina, a Resolução CFM 2.314/2022 exige consentimento explícito para compartilhar as informações do paciente, salvo emergência. A LGPD dispensa o consentimento para dados que o próprio titular tornou manifestamente públicos. Ambas partem do consentimento como regra, mas preveem exceções (emergência; dados manifestamente públicos).",
   {"resolucao_2314_CFM_2022_p7_c0": 2, "lgpd_BR_2018_c18": 2}),
 C("Como a Recomendação da UNESCO e a Estratégia Brasileira de IA tratam usos sensíveis como vigilância e reconhecimento facial?",
   "A UNESCO proíbe o uso de IA para pontuação social e vigilância em massa. A EBIA sugere salvaguardas — por exemplo, exigir que soluções de reconhecimento facial adquiridas por órgãos públicos tenham taxa de falso positivo abaixo de um limiar — e técnicas para tratar o viés. Ambas buscam limitar riscos desses usos, uma por proibição, outra por salvaguardas.",
   {"ethical_impact_assessment_UNESCO_2023_p13_c0": 2, "estrategia_brasileira_ia_MCTI_2021_p24_c1": 2}),
 C("Como o NIST AI RMF e o PL 2338/2023 lidam com a ideia de risco em sistemas de IA?",
   "O NIST destaca que riscos de IA são novos ou ampliados em relação ao software tradicional (por exemplo, dados que não representam o contexto) e exigem gestão específica. O PL 2338 adota abordagem baseada em risco: agentes de IA de alto risco devem cumprir as medidas de governança do Capítulo IV e das normas setoriais. Ambos condicionam obrigações ao nível de risco.",
   {"ai_risk_management_framework_NIST_2023_p43_c0": 2, "projeto_lei_2338_BR_2023_p14_c3": 2}),
 C("Como o PL 2338/2023 e o NIST AI RMF estruturam a governança de sistemas de IA?",
   "O PL 2338 cria uma estrutura institucional — o Sistema Nacional de Regulação e Governança de IA (SIA), coordenado pela ANPD. O NIST propõe uma estrutura funcional interna às organizações: na função MAP, requisitos como respeito à privacidade são levantados e compreendidos pelos atores relevantes. Um governa por instituições; o outro, por funções organizacionais.",
   {"projeto_lei_2338_BR_2023_p23_c0": 2, "ai_risk_management_framework_NIST_2023_p31_c1": 2}),
 C("Como a Resolução ANPD 19/2024 e o GDPR tratam a responsabilização e as sanções em proteção de dados?",
   "A ANPD 19 prevê que o juiz pode inverter o ônus da prova a favor do titular e que quem repara o dano tem direito de regresso contra os demais responsáveis. O GDPR determina que as sanções administrativas sejam efetivas, proporcionais e dissuasivas. Ambos reforçam a responsabilização, com foco na reparação ao titular e em penalidades dissuasivas.",
   {"resolucao_19_ANPD_2024_c62": 2, "gdpr_regulation_EU_2016_p28_c2": 2}),
 C("Como a FDA e o GDPR tratam a segurança dos dados em sistemas que processam dados sensíveis?",
   "A FDA recomenda, contra o envenenamento de dados em dispositivos com IA, validar/autenticar/limpar dados, detecção de anomalias, verificações de integridade e treinamento adversarial. O GDPR exige assegurar confidencialidade, integridade, disponibilidade e resiliência dos sistemas e restaurar o acesso após incidentes. Ambos exigem medidas técnicas para proteger a integridade e a disponibilidade dos dados.",
   {"ai_device_software_guidance_FDA_2025_p39_c0": 2, "gdpr_regulation_EU_2016_p52_c0": 2}),
 C("Que princípios a OMS e a UNESCO defendem para uma IA confiável?",
   "A OMS defende a responsividade: examinar contínua e transparentemente se a IA responde às necessidades do contexto, com proteção a vulneráveis contra viés e discriminação. A UNESCO enfatiza transparência, explicabilidade e responsabilização, com supervisão, auditoria e devida diligência. Ambas ligam a confiabilidade da IA à transparência e a mecanismos de correção e responsabilização.",
   {"ethics_governance_ai_health_WHO_2021_p46_c1": 2, "ethical_impact_assessment_UNESCO_2023_p36_c0": 2}),
 C("Como o Código de Ética Médica e a Lei nº 14.874/2024 asseguram a isenção de quem fiscaliza a atividade médica e a pesquisa?",
   "O Código de Ética Médica exige que o médico atue com absoluta isenção quando auditor ou perito, vedando remuneração vinculada à glosa ou ao sucesso da causa. A Lei 14.874/2024 atribui ao Comitê de Ética em Pesquisa (CEP) assegurar direitos e bem-estar dos participantes e avaliar a pesquisa de forma independente. Ambos garantem imparcialidade na supervisão.",
   {"codigo_etica_medica_CFM_2019_p41_c1": 2, "lei_14874_BR_2024_c27": 2}),
 C("Como a Resolução ANPD 19/2024 e o Código de Ética Médica atribuem responsabilidade por danos?",
   "A ANPD 19 permite ao juiz inverter o ônus da prova a favor do titular e dá à parte que reparou o dano direito de regresso contra os demais responsáveis. O Código de Ética Médica estabelece que o médico responde em caráter pessoal, e nunca presumido, por seus atos profissionais. Um enfatiza a proteção do titular e a repartição da responsabilidade; o outro, a responsabilidade pessoal do profissional.",
   {"resolucao_19_ANPD_2024_c62": 2, "codigo_etica_medica_CFM_2019_p19_c0": 2}),
 C("Como a Estratégia Brasileira de IA e o PL 2338/2023 tratam o risco de viés e discriminação em sistemas de IA?",
   "A EBIA prevê desenvolver técnicas para identificar e tratar o risco de viés algorítmico. O PL 2338 assegura à pessoa afetada o direito à não discriminação ilícita ou abusiva e à correção de vieses discriminatórios. A estratégia atua no plano técnico e de política; o projeto de lei, no plano de direitos exigíveis.",
   {"estrategia_brasileira_ia_MCTI_2021_p24_c1": 2, "projeto_lei_2338_BR_2023_p6_c1": 2}),
 C("Como a UNESCO e o NIST veem a avaliação de impacto de sistemas de IA?",
   "A UNESCO inclui a avaliação de impacto entre os mecanismos de responsabilização exigidos (junto a supervisão, auditoria e devida diligência). O NIST observa que avaliações de impacto ajudam a entender potenciais danos em contextos específicos, mas alerta para a falta de consenso sobre métricas confiáveis. Ambos valorizam a avaliação de impacto, com o NIST ressalvando os limites de medição.",
   {"ethical_impact_assessment_UNESCO_2023_p36_c0": 2, "ai_risk_management_framework_NIST_2023_p11_c0": 2}),
 C("Como o PL 2338/2023 e a Estratégia Brasileira de IA tratam as startups de inteligência artificial no Brasil?",
   "O PL 2338 determina que as autoridades setoriais definam critérios diferenciados para sistemas de IA de startups, microempresas e empresas de pequeno porte, considerando impacto concorrencial e usuários afetados. A EBIA diagnostica o ecossistema: aponta desafios das startups brasileiras (mão de obra qualificada, carga tributária, burocracia) e o investimento atraído. Um cria tratamento regulatório diferenciado; a outra descreve o ambiente.",
   {"projeto_lei_2338_BR_2023_p32_c1": 2, "estrategia_brasileira_ia_MCTI_2021_p10_c2": 2}),
 C("Como o Espaço Europeu de Dados de Saúde e a LGPD abordam o uso de dados pessoais e de saúde?",
   "O EEDS busca criar um regime jurídico e técnico uniforme para sistemas de registos de saúde eletrónicos e melhorar o acesso e o controlo das pessoas sobre seus dados de saúde. A LGPD disciplina o tratamento de dados pessoais em geral, por exemplo dispensando consentimento para dados manifestamente públicos. Ambos regulam o uso de dados, o EEDS com foco setorial em saúde e uso secundário.",
   {"european_health_data_space_EU_2025_p1_c2": 2, "lgpd_BR_2018_c18": 2}),
 C("Como a OMS e o PL 2338/2023 tratam a questão de quem deve governar a inteligência artificial?",
   "A OMS alerta que, quando grandes empresas controlam dados e algoritmos, podem passar a governar decisões que deveriam caber a indivíduos, sociedades e governos. O PL 2338 responde com governança pública: cria o SIA, coordenado pela ANPD, para regular a IA. Um aponta o risco de captura pelo setor privado; o outro institui um sistema estatal de governança.",
   {"ethics_governance_ai_health_WHO_2021_p79_c3": 2, "projeto_lei_2338_BR_2023_p23_c0": 2}),
 C("Como a FDA (dispositivos com IA) e a Resolução CFM de telemedicina tratam a qualidade e a segurança do cuidado?",
   "A FDA exige um sistema de qualidade, com controle de produtos não conformes e ações corretivas e preventivas revisadas periodicamente. A Resolução CFM 2.314/2022 considera o atendimento presencial o padrão ouro de referência e determina que as tecnologias sirvam para melhorar o exercício da medicina, preservando confidencialidade e qualidade. Ambas subordinam a tecnologia a padrões de qualidade e segurança.",
   {"ai_device_software_guidance_FDA_2025_p11_c0": 2, "resolucao_2314_CFM_2022_p9_c0": 2}),
 C("Como a OMS e o PL 2338/2023 protegem contra viés e discriminação por sistemas de IA?",
   "A OMS pede provisões especiais para proteger direitos e bem-estar de pessoas vulneráveis, com mecanismos de reparação caso surja viés ou discriminação. O PL 2338 assegura o direito à não discriminação ilícita ou abusiva e à correção de vieses discriminatórios. Ambos combinam prevenção de viés com mecanismos de correção e reparação.",
   {"ethics_governance_ai_health_WHO_2021_p46_c1": 2, "projeto_lei_2338_BR_2023_p6_c1": 2}),
 C("Como a Recomendação da UNESCO e o Código de Ética Médica tratam a autonomia da decisão humana em situações críticas?",
   "A UNESCO exige que, em decisões irreversíveis ou de vida e morte, a determinação humana final se aplique, vedando usos como a pontuação social. O Código de Ética Médica prevê que o médico aceite as escolhas do paciente sobre procedimentos (quando adequadas e reconhecidas) e, em situações terminais, evite procedimentos desnecessários e ofereça cuidados paliativos. Ambos preservam a decisão humana em contextos sensíveis.",
   {"ethical_impact_assessment_UNESCO_2023_p13_c0": 2, "codigo_etica_medica_CFM_2019_p19_c0": 2}),
 C("Como o PL 2338/2023 e a FDA tratam a governança de sistemas de IA de maior risco em setores regulados?",
   "O PL 2338 exige que agentes de IA de alto risco cumpram as medidas de governança do Capítulo IV e das legislações setoriais. A FDA, no setor de dispositivos médicos, impõe um sistema de qualidade com controle de não conformidades e ações corretivas. Ambos condicionam sistemas de IA em setores sensíveis a obrigações de governança e qualidade.",
   {"projeto_lei_2338_BR_2023_p14_c3": 2, "ai_device_software_guidance_FDA_2025_p11_c0": 2}),
 C("Como o NIST e a UNESCO propõem avaliar a gravidade ou a magnitude de impactos negativos da IA?",
   "O NIST reconhece a dificuldade de medir riscos e danos pela falta de métricas consensuais, alertando contra medições simplificadas ou manipuláveis. A UNESCO oferece uma escala qualitativa de gravidade em quatro níveis (Moderado/Menor, Sério, Crítico e Catastrófico). Um ressalta os limites da mensuração; o outro propõe uma classificação estruturada de gravidade.",
   {"ai_risk_management_framework_NIST_2023_p11_c0": 2, "ethical_impact_assessment_UNESCO_2023_p45_c0": 2}),
 C("Como o GDPR e a Resolução ANPD 19/2024 asseguram salvaguardas para os dados pessoais?",
   "O GDPR exige medidas técnicas e organizacionais — confidencialidade, integridade, disponibilidade, resiliência e restauração do acesso após incidentes. A ANPD 19 determina que as cláusulas mandatórias de transferência internacional sejam adotadas integralmente e sem alteração, para garantir a validade e as salvaguardas da transferência. Ambos fixam salvaguardas mínimas obrigatórias para proteger os dados.",
   {"gdpr_regulation_EU_2016_p52_c0": 2, "resolucao_19_ANPD_2024_c44": 2}),
 C("Como o PL 2338/2023 e o NIST AI RMF tratam o dever de informar ou notificar as pessoas sobre sistemas de IA?",
   "O PL 2338 dá à pessoa o direito à informação sobre suas interações com sistemas de IA, de forma acessível e compreensível, inclusive sobre o caráter automatizado. O NIST associa a transparência à notificação — por exemplo, informar o operador ou usuário quando um resultado adverso é detectado — e à responsabilização. Ambos exigem tornar a atuação da IA visível ao usuário.",
   {"projeto_lei_2338_BR_2023_p6_c1": 2, "ai_risk_management_framework_NIST_2023_p21_c0": 2}),
 C("Como o Espaço Europeu de Dados de Saúde e a OMS veem o uso secundário e o compartilhamento de dados de saúde?",
   "O EEDS visa facilitar o acesso e o uso secundário de dados de saúde eletrónicos (por exemplo, para pesquisa, regulação e resposta a ameaças sanitárias) sob um regime uniforme. A OMS descreve parcerias público-privadas em que dados públicos, como prontuários, são usados por empresas para desenvolver produtos, exigindo supervisão governamental. Ambos tratam do compartilhamento de dados de saúde, com atenção à governança e à supervisão.",
   {"european_health_data_space_EU_2025_p1_c2": 2, "ethics_governance_ai_health_WHO_2021_p114_c1": 2}),
 C("Que diretrizes o PL 2338/2023 e a Estratégia Brasileira de IA estabelecem para a atuação do poder público em IA?",
   "O PL 2338 traz diretrizes para a atuação da União, dos Estados e dos Municípios, como o estabelecimento de mecanismos de governança multiparticipativa. A EBIA propõe ações como definir princípios éticos de forma multissetorial, mapear barreiras legais, estimular transparência e desenvolver técnicas contra o viés. Ambos orientam o poder público a promover uma IA ética, transparente e com governança participativa.",
   {"projeto_lei_2338_BR_2023_p32_c1": 2, "estrategia_brasileira_ia_MCTI_2021_p24_c1": 2}),
]

def main():
    setup_io()
    cfg = load_config()
    corpus = load_corpus(resolve(cfg["paths"]["corpus"]))
    ids = set(corpus)
    grade_nb = cfg["golden"].get("grade_neighbor_chunk", 1)

    bad, records, seen = [], [], set()
    for q in QUESTIONS:
        for cid in q["qrels"]:
            if cid not in ids:
                bad.append((cid, q["question"][:55]))
        key = " ".join(q["question"].lower().split())
        if key in seen:
            print("  ! duplicada:", q["question"][:60]); continue
        seen.add(key)
        qrels = dict(q["qrels"])
        if q["question_type"] == "factual":            # vizinhos = grau 1
            for cid in list(qrels):
                for nb in neighbors_same_doc(cid, corpus, window=1):
                    qrels.setdefault(nb, grade_nb)
        langs = {detect_lang(corpus[c]["text"]) for c in q["qrels"] if c in ids}
        source_lang = ("en" if langs == {"en"} else "pt" if langs == {"pt"}
                       else ("mixed" if langs else None))
        src = sorted({corpus[c]["metadata"].get("source", c) for c in q["qrels"] if c in ids})
        records.append({
            "qid": f"q{len(records)+1:04d}",
            "question": q["question"].strip(),
            "reference_answer": q["reference_answer"].strip(),
            "question_type": q["question_type"],
            "source_lang": source_lang,
            "difficulty": q.get("difficulty", "medium"),
            "theme": ", ".join(sorted({corpus[c]["metadata"].get("theme", "")
                     for c in q["qrels"] if c in ids})) if q["qrels"] else "fora-de-escopo",
            "source_docs": src,
            "qrels": qrels,
        })

    if bad:
        print("✗ QRELS INVÁLIDOS (corrija antes de salvar):")
        for cid, qq in bad:
            print(f"   [{cid}]  ← {qq}")
        sys.exit(1)

    out = resolve(cfg["paths"]["golden_qa"])
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # derivados (só respondíveis)
    ans = [r for r in records if r["qrels"]]
    with open(resolve(cfg["paths"]["queries_csv"]), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["QueryId", "Query"])
        for r in ans: w.writerow([r["qid"], r["question"]])
    with open(resolve(cfg["paths"]["qrels_csv"]), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["QueryId", "ChunkId", "Relevance"])
        for r in ans:
            for cid, g in r["qrels"].items(): w.writerow([r["qid"], cid, g])

    print("por tipo:", dict(Counter(r["question_type"] for r in records)))
    print("por língua (fonte):", dict(Counter(r["source_lang"] for r in records)))
    print(f"✓ {len(records)} perguntas → {out}  ({len(ans)} respondíveis)")


if __name__ == "__main__":
    main()
