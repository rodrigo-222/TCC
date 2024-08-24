import numpy as np
import matplotlib.pyplot as plt

def lms_filter(input_signal, desired_signal, filter_order, step_size):
    # Initialize filter coefficients
    filter_coeffs = np.zeros(filter_order + 1)
    
    # Initialize filtered output
    filtered_output = np.zeros(len(input_signal))
    

    for j in range(filter_order, len(input_signal)):
            # Extract a segment of the input signal
        input_segment = input_signal[j - filter_order: j + 1][::-1]    
            # Compute the filtered output
        filtered_output[j] = np.dot(filter_coeffs, input_segment) 
            # Compute the error signal
        error = desired_signal[j] - filtered_output[j]
            # Update the filter coefficients
        filter_coeffs += step_size * error * input_segment
        
    return filtered_output, filter_coeffs

def estimate_desired_signal(input_signal, filter_coeffs):
    # Inicialize o sinal de saída
    output_signal = np.zeros_like(input_signal)

    # Para cada amostra no sinal de entrada
    for j in range(len(input_signal)):
        # Extraia um segmento do sinal de entrada
        input_segment = input_signal[j - len(filter_coeffs) + 1: j + 1][::-1]

        if len(input_segment) < len(filter_coeffs):
            input_segment = np.pad(input_segment, (len(filter_coeffs) - len(input_segment), 0))
        
        
            # Compute a saída filtrada
            output_signal[j] = np.dot(filter_coeffs, input_segment)

    return output_signal

    # Parâmetros do sinal
fs = 1000  # Frequência de amostragem
f = 5  # Frequência do sinal senoidal
t = np.arange(0, 1, 1/fs)  # Vetor de tempo
    # Exemplo de uso da função lms_filter
    # Gerar o sinal senoidal
signal = np.cos(2 * np.pi * f * t)
error_threshold = 0.01
    # Adicionar ruído gaussiano
noise = np.random.normal(0, 0.5, signal.shape)
noisy_signal = signal + noise
output_signal = noisy_signal
    # Sinal desejado (sem ruído)
desired_signal = signal

filter_order = 12  # Exemplo: ordem do filtro
step_size = 0.1  # Exemplo: taxa de aprendizado
for i in range(0, 10):
    new_output_signal, filter_coeffs = lms_filter(output_signal, desired_signal, filter_order, step_size)
    error = np.mean(np.abs(desired_signal - output_signal))
    new_error = np.mean(np.abs(desired_signal - new_output_signal))
    output_signal = new_output_signal
    if new_error < error:
        save_output_signal = new_output_signal
        save_filter_coeffs = filter_coeffs
    if error > error_threshold:
        step_size *= 0.9  # Diminui a taxa de aprendizado se o erro for alto
    else:
         step_size *= 1.1  # Aumenta a taxa de aprendizado se o erro for baixo
best_error = np.mean(np.abs(desired_signal - save_output_signal))
print("melhor error: ",best_error)
print("melhor filtro: ",save_filter_coeffs)
plt.figure(figsize=(10, 6))
plt.plot(t, save_output_signal, label='Sinal LMS', linestyle='--')
plt.plot(t, noisy_signal, label='Sinal com Ruído')
plt.plot(t, signal, label='Sinal', linestyle='-.')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Sinal Senoidal com e sem Ruído')
plt.show()
