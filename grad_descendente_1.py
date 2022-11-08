import torch
import time
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torch.autograd import Variable

# Gradiente descendente como método de otimização
# Geração de uma regressão linear com 1 variável dependente
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

# Conversão do array gerado para tensor
x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# Mudança da dimensão de y
# Ao invés de ficar na horizontal, ficará na vertical (cada campo é um registro)
y = y.view(y.shape[0], 1)

# plt.plot(x_numpy, y_numpy, 'ro')
# plt.show()


# Definição do modelo
input_size = 1 # Apenas uma variável dependente
output_size = 1
model = nn.Linear(input_size, output_size)


# Definição da função de custo e otimizador
learning_rate = 0.01
criterion = nn.MSELoss() # Método do erro quadrático médio (função de custo)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Loop de treinamento
num_epochs = 1000 # Número de interações
contador_custo = []

for epoch in range(num_epochs):
  y_hat = model(x) # y previsto
  loss = criterion(y_hat, y) # Erro
  contador_custo.append(loss)

  
  # Cálculo dos gradientes
  loss.backward()

  # Atualiação dos pesos (coeficientes)
  optimizer.step()

  if (epoch + 1) % 10 == 0:
      print('Epoch: ', epoch)
      print('Custo: {:.20f}'.format(loss.item())) 
      print('Coeficientes: ')
      print('m: {:.20f}'.format(model.weight.data.detach().item()))
      print('m (gradiente): {:.20f}'.format(model.weight.grad.detach().item()))
      print('b: {:.20f}'.format(model.bias.data.detach().item()))
      print('b (gradiente): {:.20f}'.format(model.bias.grad.detach().item()))

      previsao_final = y_hat.detach().numpy()
    #   plt.plot(x_numpy, y_numpy, 'ro') 
    #   plt.plot(x_numpy, previsao_final, 'b')
    #   plt.show()
      
    # Epoch:  999
    # Custo: 332.56756591796875000000
    # Coeficientes:
    # m: 82.48426055908203125000
    # m (gradiente): -0.00037208542926236987
    # b: 4.05405473709106445312
    # b (gradiente): 0.00002348423004150391

  # Limpeza o otimizador
  optimizer.zero_grad()