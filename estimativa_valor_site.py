from pyexpat import model
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"

df_sites = pd.read_csv(uri)


df_sites = df_sites.rename(columns={'expected_hours': 'horas_esperadas', 'price': 'preco', 'unfinished': 'nao_finalizado'})

troca_valores = {0: 1, 1: 0}

df_sites.insert(1, 'finalizado', df_sites['nao_finalizado'].map(troca_valores))


# Modelagem
x = df_sites[['horas_esperadas', 'preco']]

y = df_sites['finalizado']

seed = 131
np.random.seed(seed)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

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


# Curva de decisão
x_min = teste_x.horas_esperadas.min()

x_max = teste_x.horas_esperadas.max()

y_min = teste_x.preco.min()

y_max = teste_x.preco.max()


# Criação dos pontos para estimar os valores
pixels = 100

eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)

eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

# Coordendas
xx, yy = np.meshgrid(eixo_x, eixo_y)

# Junção das coordenadas em pares (x, y)
pontos = np.c_[xx.ravel(), yy.ravel()]


# Previsão
Z = modelo.predict(pontos)

Z = Z.reshape(xx.shape)


# Treino com nova escala
seed = 131
np.random.seed(seed)
raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

scaler = StandardScaler()

scaler.fit(raw_treino_x)

treino_x = scaler.transform(raw_treino_x)

teste_x = scaler.transform(raw_teste_x)

modelo = SVC(gamma='auto')

modelo.fit(treino_x, treino_y)

previsoes = modelo.predict(teste_x)

precisao = accuracy_score(teste_y, previsoes)
# 0.6981481481481482


# Nova curva de decisão
data_x = teste_x[:, 0] # Teste x antigo horas_esperadas

data_y = teste_x[:, 1] # Teste x antigo preço

x_min = data_x.min()

x_max = data_x.max()

y_min = data_y.min()

y_max = data_y.max()


# Criação dos pontos para estimar os valores
pixels = 100

eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)

eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

# Coordendas
xx, yy = np.meshgrid(eixo_x, eixo_y)

# Junção das coordenadas em pares (x, y)
pontos = np.c_[xx.ravel(), yy.ravel()]

Z = modelo.predict(pontos)

Z = Z.reshape(xx.shape)

ax = plt.contourf(xx, yy, Z, alpha=0.3)

ax = plt.scatter(data_x, data_y, c=teste_y, s=1).get_figure()

ax.savefig('graficos/nova_curva_de_decisao.png')
