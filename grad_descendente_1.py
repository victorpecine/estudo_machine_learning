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

plt.plot(x_numpy, y_numpy, 'ro')
plt.show()
