import numpy as np
import matplotlib.pyplot as plt

class NLMS:
    def __init__(self, num_taps, step_size, init_weights=None):
        self.num_taps = num_taps
        self.step_size = step_size
        self.weights = init_weights if init_weights is not None else np.zeros(num_taps)
        self.eps = 0.001  # Small constant to prevent divide by zero errors

    def update(self, input_vector, desired_output):
        output = np.dot(self.weights, input_vector)
        error = desired_output - output
        norm = np.dot(input_vector, input_vector) + self.eps
        self.weights += 2 * self.step_size * error * input_vector / norm
        return error

# Usage
num_taps = 10
step_size = 0.1
filter = NLMS(num_taps, step_size)

# Assume we have some input signal and desired signal
input_signal = np.random.normal(size=1000)
desired_signal = np.random.normal(size=1000)

output_signal = []

# Use o algoritmo RLS para adaptar os coeficientes do filtro
for i in range(num_taps, len(input_signal)):
    input_vector = input_signal[i-num_taps:i]
    desired_output = desired_signal[i]
    error = filter.update(input_vector, desired_output)
    # Calcula a saída do filtro para o vetor de entrada atual
    filter_output = np.dot(filter.weights, input_vector)
    # Armazena a saída do filtro
    output_signal.append(filter_output)

plt.figure(figsize=(10, 6))
plt.plot(input_signal, label='Sinal')
plt.plot(output_signal, label='Sinal Consertado', linestyle='--')
plt.plot(desired_signal, label='Sinal Desejado', linestyle='-.')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Sinal Filtrado')
plt.show()