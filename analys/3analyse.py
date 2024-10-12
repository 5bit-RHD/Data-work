
import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Функция для анализа аудиофайла с расширенными характеристиками
def analyze_audio_file(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y))  # Энергия RMS
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))  # Средний спектральный центроид
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))  # Ширина спектра
        max_amplitude = np.max(np.abs(y))  # Максимальная амплитуда
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))  # Частота пересечения нуля
        spectral_entropy = -np.sum(np.square(librosa.amplitude_to_db(y)) * np.log(np.square(librosa.amplitude_to_db(y))))  # Спектральная энтропия
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)  # Мел-частотные кепстральные коэффициенты (MFCC)
        return {
            'File': audio_file,
            'Sample Rate (Hz)': sr,
            'Duration (s)': duration,
            'RMS Energy': rms,
            'Spectral Centroid (Hz)': spectral_centroid,
            'Spectral Bandwidth (Hz)': spectral_bandwidth,
            'Max Amplitude': max_amplitude,
            'Zero-Crossing Rate': zcr,
            'Spectral Entropy': spectral_entropy,
            'MFCC': mfcc
        }
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

# Функция для анализа всех файлов в папке
def analyze_audio_folder(folder_path):
    results = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3'):
                audio_file = os.path.join(root, file)
                analysis_result = analyze_audio_file(audio_file)
                if analysis_result:
                    results.append(analysis_result)
    return results

# Сохранение результатов в CSV
def save_results_to_csv(results, output_file):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Основная функция для анализа команд и шумов
def main():
    # Путь к файлам команд и шумов (замените путь на актуальный)
    base_dir = '/Users/daniil/Хакатоны/ЦП СВФО/1/data/ржд 1/ESC_DATASET_v1.2'
    noise_folder = '/Users/daniil/Хакатоны/ЦП СВФО/1/data/ржд 1/ESC_DATASET_v1.2/luga/noise'  # папка с шумами
    output_dir = '/Users/daniil/Хакатоны/ЦП СВФО/spectograms/spectograms_new'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Анализ файлов команд
    command_folders = ['hr_bot_clear', 'hr_bot_noise', 'hr_bot_synt']
    for folder in command_folders:
        folder_path = os.path.join(base_dir, folder)
        print(f"Analyzing folder: {folder}")
        results = analyze_audio_folder(folder_path)
        save_results_to_csv(results, f'{folder}_audio_analysis.csv')
    
    # Анализ файлов шумов отдельно
    print(f"Analyzing noise folder: {noise_folder}")
    noise_results = analyze_audio_folder(noise_folder)
    save_results_to_csv(noise_results, 'noise_audio_analysis.csv')

if __name__ == "__main__":
    main()

