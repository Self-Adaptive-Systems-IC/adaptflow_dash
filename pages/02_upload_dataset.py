import streamlit as st
import requests
import os
from base64 import b64decode

st.title("Treinamentos")


st.header("Upload de dataset")

API = "http://192.168.2.131:8000"


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


# Function to load base64-encoded pickle file
def load_base64_pickle(base64_string, filename):
    """Load a base64-encoded pickle file.

    Args:
        base64_string (str): Base64-encoded string.
        filename (str): Location to save the file.
    """
    with open(filename, 'wb') as f:
        f.write(b64decode(base64_string))
        print(f"file saved on {filename}")
    

def make_upload():
    uploaded_file = st.file_uploader("Selecione um arquivo")
    result = {}
    if uploaded_file is not None:
        num_models = st.slider("Total de modelos para treinamento", 1, 15, 3)
        metric = "Accuracy"
        if st.button("Enviar"):
            file = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            params = {"n": num_models, "metric": metric}
            url = f"{API}/select_model"
            with st.spinner("Executando..."):
                response = requests.post(url, params=params, files=file)

            if response.status_code == 200:
                st.success("Arquivo enviado com sucesso!!!")
                st.json(response.json())
                result = response.json()
            else:
                st.error("Erro ao enviar o arquivo")
    
    path = './ml_models/'+uploaded_file.name
    try:
        os.mkdir(path)
        print("Folder %s created!" % path)
    except FileExistsError:
        print("Folder %s already exists" % path)
        
    for e in result['data']:
        x = e['pickle']
        load_base64_pickle(x['data'], f"{path}/{x['name']}")

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
