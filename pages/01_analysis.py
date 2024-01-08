import streamlit as st
import pandas as pd
import numpy as np
import sys
from os.path import exists

# from src.classes.AutoML import Automl
import seaborn as sns
import matplotlib.pyplot as plt

# from scipy.stats import pearsonr

st.title("Análise Exploratoria")


uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])

if uploaded_file is not None:
    dataset_df = pd.read_csv(uploaded_file)

    # Dataset
    st.subheader("Dataset")
    st.write(dataset_df.head(5))

    st.divider()

    st.subheader("Descrição")
    st.markdown(
        f"Total de **{dataset_df.shape[1]}** colunas e **{dataset_df.shape[0]}** linhas"
    )

    st.markdown(f"As colunas são **{dataset_df.columns.tolist()}**")

    st.dataframe(dataset_df.describe().T)

    st.divider()

    data_types = dataset_df.dtypes
    text_columns = data_types[data_types == "object"].index

    if text_columns.tolist() == dataset_df.columns.tolist():
        st.error(
            "Todas as colunas estão no formato de texto. É ideal converter pelo menos a coluna de target para valores numericos"
        )
        st.stop()

    with st.form("Target Column"):
        option = st.selectbox("Qual a coluna de target", dataset_df.columns.tolist())

        generator_btn = st.form_submit_button("Executar")

        if generator_btn:
            # Matriz de correlação
            st.subheader("Matriz de Correlação")
            fig_heatmap, ax_heatmap = plt.subplots()
            sns.heatmap(dataset_df.corr(), ax=ax_heatmap, annot=True)
            st.write(fig_heatmap)

            # Media das dos valores das features por target
            st.subheader("Valores médios por target")
            target_values = dataset_df[option].unique()
            mean_per_taget = dataset_df.groupby(option).mean()
            fig, ax = plt.subplots(nrows=1, ncols=len(target_values), figsize=(5, 5))

            for val in mean_per_taget.index:
                st.write(val)
                plt.subplot(1, 2, val + 1)
                sns.heatmap(
                    mean_per_taget.loc[val, :].to_frame(),
                    annot=True,
                    linewidths=0.4,
                    linecolor="black",
                    cbar=False,
                    fmt=".2f",
                )
            fig.tight_layout(pad=2)
            st.write(fig)

            st.divider()
