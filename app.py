import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import dataset as ds
from environment import Environment

def main():
    st.title("Solucionador de escalonamento FJSP")

    # Configuração do projeto
    st.sidebar.subheader("Projeto")
    proj_params = st.sidebar.radio("Selecionar Projeto:",("Novo projeto", "Abrir projeto"))
    if proj_params == "Novo projeto":
        pass
    elif proj_params == "Abrir projeto":
        pass


    # Configuração dos parâmetros do ambiente(dataset)
    st.sidebar.subheader("Parâmetros do Ambiente")
    env_params = st.sidebar.radio("Selecionar dataset:",("Selecionar exemplo", "Abrir arquivo"))

    if env_params == "Selecionar exemplo":
        dataset_options = ds.list_files()
        selectbox_dataset = st.sidebar.selectbox("Dataset", dataset_options)
        file = ds.read_file(selectbox_dataset)
        env = Environment(file)
        for i in env.print_jobs:
            st.text(i)

    elif env_params == "Abrir arquivo":
        pass


    # Configuração dos parâmetros do agente
    st.sidebar.subheader("Parâmetros do Agente")
    alpha = st.sidebar.slider("Alpha", 0, 100, 50)
    gamma = st.sidebar.slider("Gamma", 0, 100, 50)
    n_actions = st.sidebar.slider("Número de ações", 0, 100, 50)
    epsilon = st.sidebar.slider("Epsilon", 0, 100, 50)
    

if __name__ == '__main__':
    main()