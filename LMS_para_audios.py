import os
import numpy as np
import librosa
import soundfile as sf

def lms_filter(input_signal, desired_signal, filter_order, step_size):
    filter_coeffs = np.zeros(filter_order + 1)
    filtered_output = np.zeros(len(input_signal))
    for j in range(filter_order, len(input_signal)):
        input_segment = input_signal[j - filter_order: j + 1][::-1]
        filtered_output[j] = np.dot(filter_coeffs, input_segment)
        error = desired_signal[j] - filtered_output[j]
        filter_coeffs += step_size * error * input_segment
    return filtered_output, filter_coeffs

def load_audio_file(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    return signal, sr

def process_audio_files(input_dir, output_dir, desired_dir):
    for filename, filename_disired in zip(os.listdir(input_dir),os.listdir(desired_dir)):
        if filename == filename_disired:
            file_path = os.path.join(input_dir, filename)
            file_path_disired = os.path.join(desired_dir, filename_disired)
            input_signal, fs = load_audio_file(file_path)
            desired_signal, fs = load_audio_file(file_path_disired)
            print(f'Processing file: {filename}')
            print(f'Processing file: {filename_disired}')
            # Aplicar o filtro LMS
            filter_order = 12
            step_size = 0.1
            output_signal, filter_coeffs = lms_filter(input_signal, desired_signal, filter_order, step_size)
            print(f'Filter coefficients: {filter_coeffs}')
            # Salvar o áudio filtrado
            output_file_path = os.path.join(output_dir, filename)
            sf.write(output_file_path, output_signal, fs)

# Caminhos dos diretórios de entrada e saída
input_dir = os.path.expanduser('~/Documentos/TCC/audios_gauseanos')
desired_dir = os.path.expanduser('~/Documentos/TCC/audios')
output_dir = os.path.expanduser('~/Documentos/TCC/audio_corrigido')

# Processar os arquivos
process_audio_files(input_dir, output_dir, desired_dir)