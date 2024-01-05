import streamlit as st
import requests

st.title("Treinamentos")


st.header("Upload de dataset")

API = "http://localhost:8000"


def check_if_api():
    try:
        response = requests.get(url=API)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"Status": response.status_code}
    except requests.ConnectionError as e:
        return False, {"Status": e}
    response = requests.get(url=API)
    st.write(response.status_code)
    # st.write(response.json())


api_status, api_message = check_if_api()


def make_upload():
    uploaded_file = st.file_uploader("Selecione um arquivo")

    if uploaded_file is not None:
        num_models = st.slider("Total de modelos para treinamento", 1, 15, 3)
        metric = "Accurecy"
        if st.button("Enviar"):
            file = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            params = {"n": num_models, "metric": metric}
            url = f"{API}/select_model"
            with st.spinner("Executando..."):
                response = requests.post(url, params=params, files=file)

            if response.status_code == 200:
                st.success("Arquivo enviado com sucesso!!!")
                st.json(response.json())
            else:
                st.error("Erro ao enviar o arquivo")


if not api_status:
    st.markdown(
        f"""
        This application could not establish a connection to the API.
        Please make sure that the API is running and acessible.
        Check the API endpoint, API url, network connection or server status.

        Message error **{api_message['Status']}**
        """
    )
elif api_status and api_message["Status"] != "Connected":
    st.markdown(
        f"""
        This application could establish a connection to the API,
        but the API is {api_message["Status"]}
        """
    )
else:
    make_upload()
