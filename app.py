import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from agent.agent import graph
from agent.utils.tools import retrieve_information


st.set_page_config(
    page_title="NIAR Chat",
    page_icon="🧠"
)

st.title("🧠 Chat NIAR")
st.write("Assistente baseado nos documentos do NIAR")


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)

    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)


prompt = st.chat_input("Digite sua pergunta...")


if prompt:
    human_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(human_msg)

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando base de documentos..."):

            # 1. Recupera documentos no Qdrant
            contexto = retrieve_information.invoke({"query": prompt})

            # 2. Monta uma pergunta já com contexto obrigatório
            pergunta_com_contexto = f"""
Use exclusivamente os documentos recuperados abaixo para responder.

Pergunta do usuário:
{prompt}

Documentos recuperados:
{contexto}

Instruções:
- Responda com base nos documentos recuperados.
- Se os documentos não forem suficientes, diga isso claramente.
- Ao final, inclua uma seção "Fontes utilizadas".
"""

            # 3. Chama o agente com o contexto já recuperado
            result = graph.invoke({
                "messages": [HumanMessage(content=pergunta_com_contexto)],
                "confirmation": False,
                "route": None,
                "user_data": None,
                "debug": None,
            })

            resposta = result["messages"][-1].content

            st.markdown(resposta)

            st.session_state.messages.append(
                AIMessage(content=resposta)
            )
