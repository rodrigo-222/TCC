import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do ruído gaussiano
mu = 0  # média
sigma = 1  # desvio padrão

# Gerar ruído gaussiano
noise = np.random.normal(mu, sigma, 1000)

# Plotar histograma
plt.hist(noise, bins=30, density=True, alpha=0.6, color='g')

# Adicionar curva da distribuição normal
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
plt.plot(x, p, 'k', linewidth=2)
title = "Histograma do Ruído Gaussiano"
plt.title(title)
plt.show()
