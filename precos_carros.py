import pandas as pd
from datetime import datetime


uri = "https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv"

df_carros = pd.read_csv(uri)

df_carros = df_carros.rename(columns={'mileage_per_year': 'milhas_por_ano',
                                      'model_year': 'ano_do_modelo',
                                      'price': 'preco',
                                      'sold':'vendido'})

trocar_vendido = {'no': 0, 'yes': 1}

df_carros['vendido'] = df_carros['vendido'].map(trocar_vendido)



ano_atual = datetime.today().year

df_carros.insert(3, 'idade_do_modelo', ano_atual - df_carros['ano_do_modelo'])


df_carros['km_por_ano'] = df_carros['milhas_por_ano'] * 1.60934


df_carros = df_carros.drop(columns=['Unnamed: 0', 'milhas_por_ano'], axis=1)
