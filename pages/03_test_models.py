import streamlit as st
import requests
import pandas as pd
import random
import os
import json
import time
import plotly.express as px  # interactive charts

API_URL = "http://localhost"
HISTORICAL_DATA = "data_dataset_edit_01.csv"


st.set_page_config(
    page_title="Dash",
    page_icon="📊",
    layout="wide",
)

# TODO check if have anny prediction api running
# TODO Make a away to use the first api to get the features


def check_api(port):
    url = f"{API_URL}:{port}/"  # Substitua pela URL da rota base da sua API
    try:
        response = requests.get(url)
        # st.write(response.json())
        if response.status_code == 200:
            return True, response.json()
        else:
            # st.toast(f"Non-200 status code recived: {response.status_code}")
            return False, {}
    except requests.ConnectionError as e:
        # st.toast(f"Connection error: {e}")
        return False, {}


def scan_running_apis():
    result_dict = {}
    for port in range(5000, 5010):
        test, res = check_api(port)
        # st.write(res)
        if res:
            model_name = res["model"]  # Assumindo que `res` contém o nome do modelo
            result_dict[model_name] = port
    return result_dict


def generate_new_param(features):
    params = {}
    for key, value in features.items():
        # st.write(column,x)
        params[key] = random.uniform(value[0], value[1])
    return params


def requests_2_apis(model: str, time: int, data: dict, df):
    url = f"http://127.0.0.1:{running_apis[model]}/predict"
    response = requests.post(url, data=json.dumps(data))
    # response = requests.post(routes[model], data=json.dumps(data))
    result = response.json()
    # print(result)

    # b = {"acc": result["prediction"], "model": model}

    infos = {
        "model": model,
        "response": result["prediction"],
        "acc": result["score"] * 100,
    }
    return infos


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


# Busca as apis que estão em execução
running_apis = scan_running_apis()
st.write("Running Apis")
running_apis_df = pd.DataFrame(list(running_apis.items()), columns=["Modelo", "Porta"])
st.write(running_apis_df)

# Busca os nomes dos modelos em execução
model_names = list(running_apis)

# dashboard title
st.markdown("# Teste dos modelos")

# Buscas as features (Pensar numa forma melhor isso aqui)
response = requests.get(f"{API_URL}:5000/get_features")
print(response)
features = response.json()
# st.write(features)

# Define os placeholders necessarios
placeholder_form = st.empty()
placeholder_data_visualization = st.empty()


def test_model(model_name):
    # Define as variaveis necessarias para realziar os testes
    last_acc = 0  # Pegar a acc max antes de inserir novas
    all_results = historical_data_df  # Copia os dados historicos
    max_lenght = (
        0 if all_results.shape[0] == 0 else all_results.iloc[-1]["time"] + 1
    )  # Pega o maior tempo dentro do dataset
    model_acc_mean_df = pd.DataFrame(
        columns=["time", "mean"]
    )  # Dataframe para gerar os valores médios do modelo selecionado
    df_2 = pd.DataFrame()
    # Loop para realizar consultas nos modelos
    for seconds in range(0, 2000):
        with placeholder_data_visualization.container():
            elapsed_time = seconds + max_lenght  # Total de tempo passado
            new_data = generate_new_param(
                features
            )  # Gera um novo conjunto de dados para teste

            # Para cada modelo vai realizar a consuta na respectiva api
            info = {}
            for model in model_names:
                info[model] = requests_2_apis(
                    model, elapsed_time, new_data, historical_data_df
                )

            # Preenche o dataframe com os resultados obtidos das predições
            historical_data = []
            for model, model_infos in info.items():
                historical_data.append(
                    {
                        "model": model,
                        "accuracy": model_infos["acc"],
                        "time": elapsed_time,
                    }
                )
            all_results = pd.concat(
                [all_results, pd.DataFrame(historical_data)], ignore_index=True
            )
            all_results.to_csv(f"./tmp/historical_data/{HISTORICAL_DATA}", index=False)

            # Filta os dasos historicos do modelo selecionado
            filtered_info = info[option]

            # Define um dataframe com os valores médios por tempo
            all_model_op_results = all_results[
                all_results["model"] == option
            ]  # Filtra os resultados historicos do modelo
            model_acc_mean = all_model_op_results[
                "accuracy"
            ].mean()  # Calcula a média da acuracia do modelo

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

            # st.write(new_data)
            time.sleep(0.1)


model_name = "Ada_Boost_Classifier"

with placeholder_form.container():
    with st.form("form1"):
        st.write("Selecione um modelo")
        # st.markdown(f"**Modelo atual**: {model_name}")
        option = st.selectbox("Modelo", options=(model_names), key="visibility")
        st.write(option)
        submitted = st.form_submit_button("Submit")
        if submitted:
            historical_data_df.to_csv(HISTORICAL_DATA, index=False)
            model_name = option
            test_model(model_name)
