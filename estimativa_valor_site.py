import pandas as pd
import seaborn as sns


uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"

df_sites = pd.read_csv(uri)


df_sites = df_sites.rename(columns={'expected_hours': 'horas_esperadas', 'price': 'preco', 'unfinished': 'nao_finalizado'})

troca_valores = {0: 1, 1: 0}

df_sites.insert(1, 'finalizado', df_sites['nao_finalizado'].map(troca_valores))


# Análise gráfica
ax = sns.scatterplot(data=df_sites, x='horas_esperadas', y='preco', hue='finalizado').get_figure()

ax.savefig('graficos/horas_preco.png')


ax_1 = sns.relplot(data=df_sites, x='horas_esperadas', y='preco', col='finalizado', hue='finalizado')

ax_1.savefig('graficos/horas_preco_colunas.png')
