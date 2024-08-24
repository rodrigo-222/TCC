import numpy as np
import matplotlib.pyplot as plt

class RLS:
    def __init__(self, num_taps, forgetting_factor, init_weights=None):
        self.num_taps = num_taps
        self.forgetting_factor = forgetting_factor
        self.weights = init_weights if init_weights is not None else np.zeros(num_taps)
        self.inverse_correlation_matrix = np.eye(num_taps)

    def update(self, input_vector, desired_output):
        prediction = np.dot(self.weights, input_vector)
        prediction_error = desired_output - prediction
        gain = (self.inverse_correlation_matrix @ input_vector) / (self.forgetting_factor + input_vector.T @ self.inverse_correlation_matrix @ input_vector)
        self.weights += gain * prediction_error
        self.inverse_correlation_matrix = (self.inverse_correlation_matrix - np.outer(gain, input_vector.T @ self.inverse_correlation_matrix)) / self.forgetting_factor
        return prediction_error

# Usage
num_taps = 10
forgetting_factor = 0.99
filter = RLS(num_taps, forgetting_factor)

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