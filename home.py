import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon=":large_purple_circle:",
)



st.write("# Dashboard XXXXX")

st.markdown(
    """
    ### Bem Vindo!
    
    Essa é a dashboard para auxiliar no entendimento e funcionamento do serviço XXXX.
    Aqui você pode gerar uma [análise exploratoria](/analysis) de seu conjunto de dados, realizar o
    [treinamento](/upload_dataset) de "n" modelos de aprendizado de máquina e realizar
    [testes](/test_models) nos modelos gerados.
    
    Para mais infomações sobre o serviço acesse: 
    - [GitHub Repositorio (API)](https://github.com/Self-Adaptive-Systems-IC/adaptflow_api)
    - [GitHub Repositorio (Dashboard)](https://github.com/Self-Adaptive-Systems-IC/adaptflow_api)
    
    Para exemplos de como integrar o serviço em seu sistema acesse:
    - LINK AQUI
"""
)