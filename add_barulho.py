import os
import numpy as np
import soundfile as sf

def add_gaussian_noise(directory, directory2):
    # Get all audio files in the directory
    audio_files = [file for file in os.listdir(directory) if file.endswith('.mp3')]

    for file in audio_files:
        # Load the audio file
        audio, sample_rate = sf.read(os.path.join(directory, file))

        # Generate Gaussian noise with the same shape as the audio
        noise = np.random.normal(0, 0.1, audio.shape)

        # Add the noise to the audio
        noisy_audio = audio + noise

        # Save the modified audio file
        sf.write(os.path.join(directory2, f'{file}'), noisy_audio, sample_rate)

    # Return the modified audio files


add_gaussian_noise("/home/roger/Documentos/TCC/audios", "/home/roger/Documentos/TCC/audios_gauseanos")