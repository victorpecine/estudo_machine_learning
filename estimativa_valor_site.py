import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np


uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"

df_sites = pd.read_csv(uri)


df_sites = df_sites.rename(columns={'expected_hours': 'horas_esperadas', 'price': 'preco', 'unfinished': 'nao_finalizado'})

troca_valores = {0: 1, 1: 0}

df_sites.insert(1, 'finalizado', df_sites['nao_finalizado'].map(troca_valores))


# Modelagem
x = df_sites[['horas_esperadas', 'preco']]

y = df_sites['finalizado']

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state=131, test_size=0.25, stratify=y)

modelo = LinearSVC()

modelo.fit(treino_x, treino_y)

previsoes = modelo.predict(teste_x)

precisao = accuracy_score(teste_y, previsoes)
# 0.5370370370370371


# Previsão de base considerando todos os projetos como finalizados
previsao_de_base = np.ones(540)

precisao_de_base = accuracy_score(teste_y, previsao_de_base)
# 0.5259259259259259


# O modelo estimado tem uma precisao 2,11% maior que a previsão de base
# É necessário aumentar a precisão do modelo