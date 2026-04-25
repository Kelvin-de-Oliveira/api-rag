import requests
import streamlit as st
from streamlit_chat import message


DEFAULT_API_URL = "http://localhost:8000"


def formatar_fontes(fontes: list[dict]) -> str:
    if not fontes:
        return "Sem fontes retornadas pela API."

    fontes_unicas: dict[tuple[str, str, str], dict] = {}

    for fonte in fontes:
        id_processo = (fonte.get("id_processo") or "").strip()
        data_publicacao = (fonte.get("data_publicacao") or "").strip()
        url = (fonte.get("url") or "").strip()
        indices_citacao = fonte.get("indices_citacao") or []

        chave = (id_processo.lower(), data_publicacao, url.lower())

        if chave not in fontes_unicas:
            fontes_unicas[chave] = {
                "id_processo": id_processo,
                "data_publicacao": data_publicacao,
                "url": url,
                "indices_citacao": [],
            }

        indices_existentes = fontes_unicas[chave]["indices_citacao"]
        for indice in indices_citacao:
            if indice not in indices_existentes:
                indices_existentes.append(indice)

    linhas: list[str] = []
    for idx, fonte in enumerate(fontes_unicas.values(), start=1):
        id_processo = fonte["id_processo"]
        data_publicacao = fonte["data_publicacao"]
        url = fonte["url"]
        indices_citacao = fonte["indices_citacao"]

        linhas.append(f"{idx}.")
        if id_processo:
            linhas.append(f"- Processo: {id_processo}")
        if data_publicacao:
            linhas.append(f"- Data: {data_publicacao}")
        if url:
            linhas.append(f"- URL: {url}")
        if indices_citacao:
            indices_formatados = ", ".join(f"[{indice}]" for indice in indices_citacao)
            linhas.append(f"- Indices de citacao: {indices_formatados}")

    return "\n".join(linhas)


def consultar_rag(pergunta: str, api_url: str, timeout_s: int) -> dict:
    endpoint = f"{api_url.rstrip('/')}/perguntar"
    resposta = requests.post(
        endpoint,
        json={"pergunta": pergunta},
        timeout=timeout_s,
    )
    resposta.raise_for_status()
    return resposta.json()


def on_input_change() -> None:
    user_input = st.session_state.user_input.strip()
    if not user_input:
        return

    st.session_state.past.append(user_input)

    try:
        with st.spinner("Consultando o RAG..."):
            resultado = consultar_rag(
                pergunta=user_input,
                api_url=st.session_state.api_url,
                timeout_s=st.session_state.timeout_s,
            )

        resposta_bot = {
            "resposta": resultado.get("resposta", "Resposta vazia da API."),
            "fontes": resultado.get("fontes", []),
            "tempo": resultado.get("tempo_processamento_s"),
            "erro": False,
        }
    except requests.HTTPError as exc:
        detalhe = ""
        if exc.response is not None:
            try:
                detalhe = exc.response.json().get("detail", "")
            except ValueError:
                detalhe = exc.response.text
        resposta_bot = {
            "resposta": f"Erro HTTP ao consultar API: {detalhe or str(exc)}",
            "fontes": [],
            "tempo": None,
            "erro": True,
        }
    except requests.RequestException as exc:
        resposta_bot = {
            "resposta": (
                "Falha de conexao com a API RAG. "
                "Verifique se ela esta rodando e se a URL esta correta.\n\n"
                f"Detalhe tecnico: {exc}"
            ),
            "fontes": [],
            "tempo": None,
            "erro": True,
        }

    st.session_state.generated.append(resposta_bot)
    st.session_state.user_input = ""


def on_btn_click() -> None:
    st.session_state.past = []
    st.session_state.generated = []


st.session_state.setdefault("past", [])
st.session_state.setdefault("generated", [])
st.session_state.setdefault("api_url", DEFAULT_API_URL)
st.session_state.setdefault("timeout_s", 60)

st.set_page_config(page_title="Chat RAG ANEEL", page_icon="💬", layout="centered")
st.title("Chat RAG ANEEL")
st.caption("Interface local para conversar com a API RAG.")

with st.sidebar:
    st.subheader("Configuracao da API")
    st.text_input("URL base da API", key="api_url", help="Ex.: http://localhost:8000")
    st.number_input("Timeout (segundos)", min_value=5, max_value=300, key="timeout_s")
    st.button("Limpar conversa", on_click=on_btn_click, use_container_width=True)

chat_placeholder = st.container()

with chat_placeholder:
    for i in range(len(st.session_state.generated)):
        pergunta = st.session_state.past[i]
        resposta = st.session_state.generated[i]

        message(pergunta, is_user=True, key=f"{i}_user")
        message(resposta["resposta"], key=f"{i}")

        if resposta.get("tempo") is not None:
            st.caption(f"Tempo de processamento: {resposta['tempo']:.2f}s")

        if resposta.get("fontes"):
            with st.expander("Fontes utilizadas"):
                st.markdown(formatar_fontes(resposta["fontes"]))

        st.divider()

with st.container():
    st.text_input("Sua pergunta", on_change=on_input_change, key="user_input")
