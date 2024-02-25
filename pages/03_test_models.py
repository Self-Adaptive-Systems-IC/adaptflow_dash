import streamlit as st
import requests
import pandas as pd
import random
import os
import json
import time
import plotly.express as px  # interactive charts
import joblib
import numpy as np
API_URL = "http://localhost"
HISTORICAL_DATA = "data_dataset_edit_01.csv"


st.set_page_config(
    page_title="Dash",
    page_icon="📊",
    layout="wide",
)

# Pasta onde será salvo os arquivos dos modelos
folder_path = "./ml_models"

# Função para listar pastas dentro de um pasta
def list_folders(path):
    return [f.path for f in os.scandir(path) if f.is_dir()]

# Função para listar os arquivos dentro de uma pasta
def list_files_in_folder(folder_path):
    return [f.name for f in os.scandir(folder_path) if f.is_file()]

# Lista de pastas (Cada dataset é salvo em uma pasta)
folders = list_folders(folder_path)

# Seleciona o primeiro dataset para aparecer no menu dropdown
option_dataset = folders[0]

# Array para armazenar os nomes dos modelos
model_names = []

# Gera parametros aleatorios para testar os modelos de ML
def generate_new_param(features):
    params = {}
    for key, value in features.items():
        # st.write(column,x)
        params[key] = random.uniform(value[0], value[1])
    return params

# Realiza a predição e calcula a porcentagem de acerto
def predict_models(model, data: dict):
    numerical_data = np.array(list(data.values())).reshape(1,-1)
    prediction = model.predict(numerical_data)[0]
    probabilities = model.predict_proba(numerical_data)[0]
    score = probabilities[prediction]
    
    infos = {
        "model": model.__class__.__name__,
        "response": prediction,
        "acc": score * 100,
    }
    
    return infos

# Carrega o dataset historico
def get_data(filename) -> pd.DataFrame:
    file_path = "./tmp/historical_data"
    file = f"{file_path}/{filename}"
    if os.path.exists(file):
        dataset = pd.read_csv(file)
    else:
        dataset = pd.DataFrame()

    return dataset


# Busca os dados historicos de consultas na api
historical_data_df = get_data(HISTORICAL_DATA)


# dashboard title
st.markdown("# Teste dos modelos")

# Buscas as features (Pensar numa forma melhor isso aqui)
# response = requests.get(f"{API_URL}:5000/get_features")
# print(response)
features = {'type_of_failure': (1, 10), 'time_repair': (-0.6304655464, 1.977824924), 'cost': (-0.843, 1.712), 'criticality': (0.0, 0.912), 'humid': (5, 85), 'temp': (24, 150)}
# st.write(features)

# Define os placeholders necessarios
placeholder_form = st.empty()
placeholder_data_visualization = st.empty()

# Realiza os graficos
def test_model(actual_model,ml_models):
    # Define as variaveis necessarias para realziar os testes
    # Pegar a acc max antes de inserir novas
    last_acc = 0
    
    # Copia os dados historicos
    all_results = historical_data_df
    
    # Pega o maior tempo dentro do dataset
    max_lenght = (
        0 if all_results.shape[0] == 0 else all_results.iloc[-1]["time"] + 1
    )
    
    # Dataframe para gerar os valores médios do modelo selecionado
    model_acc_mean_df = pd.DataFrame(
        columns=["time", "mean"]
    )  
    
    # Cria um dataframe vazio
    df_2 = pd.DataFrame()
    
    # Loop para realizar consultas nos modelos
    for seconds in range(0, 10000):
        with placeholder_data_visualization.container():
            # Total de tempo passado
            elapsed_time = seconds + max_lenght  
            
            # Gera um novo conjunto de dados para teste
            new_data = generate_new_param(features)  

            # Para cada modelo vai realizar a consuta na respectiva api
            info = {}
            for file,model in ml_models:
                info[file] = predict_models(model, new_data)

            # Preenche o dataframe com os resultados obtidos das predições
            historical_data = []
            for model, model_infos in info.items():
                historical_data.append(
                    {
                        "model": model_infos['model'],
                        "accuracy": model_infos["acc"],
                        "time": elapsed_time,
                    }
                )
            all_results = pd.concat(
                [all_results, pd.DataFrame(historical_data)], ignore_index=True
            )
            all_results.to_csv(f"./tmp/historical_data/{HISTORICAL_DATA}", index=False)

            # Filta os dasos historicos do modelo selecionado
            filtered_info = info[option_model]

            # Define um dataframe com os valores médios por tempo
            
            # Filtra os resultados historicos do modelo
            all_model_op_results = all_results[
                all_results["model"] == actual_model
            ]  
            
            # Calcula a média da acuracia do modelo
            model_acc_mean = all_model_op_results[
                "accuracy"
            ].mean()  

            model_acc_mean_df = pd.concat(
                [
                    model_acc_mean_df,
                    pd.DataFrame([{"time": elapsed_time, "mean": model_acc_mean}]),
                ],
                ignore_index=True,
            )

            df_2 = pd.concat(
                [
                    df_2,
                    pd.DataFrame(
                        [
                            {
                                "status": filtered_info["response"],
                                "acertos": filtered_info["acc"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            df_3 = df_2.groupby(by=["status"]).size().reset_index(name="counts")

            # ============================================================================
            # Exibe o nome do modelo, acuracia e resultado da predição
            # ============================================================================
            # st.subheader(f"""**Model**: {filtered_info['model']} $\Rightarrow$ *Resultado*: {filtered_info['response']} **({filtered_info['acc']}%)**""")
            model_title, acc_title, predict_title = st.columns(3)

            with model_title:
                st.header(filtered_info["model"])
            with acc_title:
                st.metric(
                    label="Accuracy",
                    value=f"{round(filtered_info['acc'],2)}%",
                    delta=f"{round((filtered_info['acc']-last_acc),2)}%",
                )
                last_acc = filtered_info["acc"]
            with predict_title:
                st.metric(label="Prediction", value=filtered_info["response"])

            # ============================================================================
            # Exibe a precisão média do modelo
            # ============================================================================
            st.markdown("### Precisão do modelo selecionado")
            fig_model_acc_mean_line, fig_model_acc_mean_dots = st.columns(2)

            with fig_model_acc_mean_line:
                st.markdown("### Precisão Média - Linha")
                fig_acc01 = px.line(
                    data_frame=model_acc_mean_df,
                    x="time",
                    y="mean",
                    # color="model",
                    range_y=[50, 100],
                    range_x=[max(0, elapsed_time - 10), elapsed_time + 2],
                )
                st.write(fig_acc01)

            with fig_model_acc_mean_dots:
                st.markdown("### Precisão Média - Pontos")
                fig2 = px.scatter(
                    data_frame=model_acc_mean_df,
                    x="time",
                    y="mean",
                    # color="model",
                    range_y=[50, 100],
                    range_x=[max(0, elapsed_time - 10), elapsed_time + 2],
                )
                st.write(fig2)

            # ============================================================================
            # Exibe a variação dos acertos para todos os modelos
            # ============================================================================
            st.markdown("### Precisão da classificação dos modelos")

            fig_acc_models_line, fig_acc_models_dots = st.columns(2)
            with fig_acc_models_line:
                st.markdown("### Acuracia do Resultado - Linha")
                fig1 = px.line(
                    data_frame=all_results,
                    x="time",
                    y="accuracy",
                    color="model",
                    range_y=[0, 100],
                    range_x=[max(0, elapsed_time - 50), elapsed_time + 2],
                )
                # st.line_chart(data=df, x="time", y="accuracy", use_container_width=True)
                st.write(fig1)

            with fig_acc_models_dots:
                st.markdown("### Acuracia do Resultado - Pontos")
                fig2 = px.scatter(
                    data_frame=all_results,
                    x="time",
                    y="accuracy",
                    color="model",
                    range_y=[0, 100],
                    range_x=[max(0, elapsed_time - 50), elapsed_time + 2],
                )
                st.write(fig2)

            # ============================================================================
            # Exibe a média de arcertos
            # ============================================================================

            fig_a, fig_b = st.columns(2)
            with fig_a:
                st.markdown("### Porcentagem da classificação entre os targets")
                fig3 = px.pie(
                    data_frame=df_2, values="acertos", names="status", color="status"
                )
                st.write(fig3)

            with fig_b:
                st.markdown("### Quantitativo por target")
                fig4 = px.bar(
                    data_frame=df_3,
                    x="status",
                    y="counts",
                    color="status",
                    barmode="group",
                )
                st.write(fig4)
            
            time.sleep(0.1)

with placeholder_form.container():
    with st.form("select_dataset"):
        st.write("Selecione um dataset")
        option_dataset = st.selectbox("Dataset", options=(folders),key="visibility")
        option_model = st.selectbox("Dataset", options=(list_files_in_folder(option_dataset)))
        submit_dataset = st.form_submit_button("Selecionar")
        if submit_dataset:
            ml_models = [(e,joblib.load(f"{option_dataset}/{e}")) for e in list_files_in_folder(option_dataset)]
            historical_data_df.to_csv(HISTORICAL_DATA, index=False)
            actual_model = ''
            for item in ml_models:
                if item[0] == option_model:
                    actual_model = item[1].__class__.__name__
            test_model(actual_model,ml_models)
