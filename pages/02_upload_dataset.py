import streamlit as st
import requests

st.title("Treinamentos")


st.header("Upload de dataset")

API = "http://localhost:8000"


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
