import numpy as np
import matplotlib.pyplot as plt

def generate_pink_noise(N):
    # Número de pontos para o ruído rosa
    uneven = N % 2
    X = np.random.randn(N // 2 + 1 + uneven) + 1j * np.random.randn(N // 2 + 1 + uneven)
    S = np.arange(len(X)) + 1  # Escala 1/f
    y = (np.fft.irfft(X / S)).real
    
    if uneven:
        y = y[:-1]
    return y

# Parâmetros
N = 1024  # Número de amostras

# Gerar ruído rosa
pink_noise = generate_pink_noise(N)

# Plotar o sinal de ruído rosa
plt.figure(figsize=(10, 6))
plt.plot(pink_noise, color='r')
plt.title("Sinal de Ruído Rosa")
plt.xlabel("Amostras")
plt.ylabel("Amplitude")
plt.show()

# Plotar o espectro de potência
plt.figure(figsize=(10, 6))
plt.psd(pink_noise, NFFT=512, Fs=1, color='r')
plt.title("Densidade Espectral de Potência do Ruído Rosa")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Potência/Frequência (dB/Hz)")
plt.show()
