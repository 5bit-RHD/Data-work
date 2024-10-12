
import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display

# Функция для анализа аудиофайла
def analyze_audio_file(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y))  # Энергия RMS
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))  # Средний спектральный центроид
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))  # Ширина спектра
        max_amplitude = np.max(np.abs(y))  # Максимальная амплитуда
        return {
            'File': audio_file,
            'Sample Rate (Hz)': sr,
            'Duration (s)': duration,
            'RMS Energy': rms,
            'Spectral Centroid (Hz)': spectral_centroid,
            'Spectral Bandwidth (Hz)': spectral_bandwidth,
            'Max Amplitude': max_amplitude
        }
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

# Функция для построения спектрограммы
def plot_spectrogram(audio_file, output_dir):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram for {os.path.basename(audio_file)}')
        plt.tight_layout()
        spectrogram_path = os.path.join(output_dir, f'spectrogram_{os.path.basename(audio_file)}.png')
        plt.savefig(spectrogram_path)
        plt.close()
        print(f"Spectrogram saved at {spectrogram_path}")
    except Exception as e:
        print(f"Error creating spectrogram for {audio_file}: {e}")

# Функция для анализа всех файлов в папке
def analyze_audio_folder(folder_path, output_dir):
    results = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3'):
                audio_file = os.path.join(root, file)
                analysis_result = analyze_audio_file(audio_file)
                if analysis_result:
                    results.append(analysis_result)
                    plot_spectrogram(audio_file, output_dir)  # Сохраняем спектрограмму для каждого файла
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
    output_dir = '/Users/daniil/Хакатоны/ЦП СВФО/spectograms'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Анализ команд (все папки команд)
    command_folders = ['hr_bot_clear', 'hr_bot_noise', 'hr_bot_synt', 'luga']
    for folder in command_folders:
        folder_path = os.path.join(base_dir, folder)
        print(f"Analyzing folder: {folder}")
        results = analyze_audio_folder(folder_path, output_dir)
        save_results_to_csv(results, f'{folder}_audio_analysis.csv')

if __name__ == "__main__":
    main()

