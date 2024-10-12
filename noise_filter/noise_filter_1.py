import os
import librosa
import soundfile as sf
import numpy as np
import scipy.signal as signal
from scipy.signal import wiener

# Полосовой фильтр для удаления ненужных частот
def bandpass_filter(data, lowcut, highcut, sr, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data

# Адаптивный Wiener фильтр для удаления шума
def apply_wiener_filter(data):
    return wiener(data)

# Основная функция для очистки аудио
def clean_audio(audio_file, output_file, lowcut=100, highcut=8000):
    try:
        # Загрузка аудиофайла
        y, sr = librosa.load(audio_file, sr=None)
        
        # Применение полосового фильтра
        filtered_data = bandpass_filter(y, lowcut, highcut, sr)
        
        # Применение Wiener фильтра для адаптивного удаления шума
        cleaned_data = apply_wiener_filter(filtered_data)
        
        # Сохранение очищенного файла
        sf.write(output_file, cleaned_data, sr)
        print(f"Cleaned audio saved as {output_file}")
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")

# Функция для обработки всех файлов в папке
def process_all_files(audio_files, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        output_file = os.path.join(output_dir, 'cleaned_' + filename)
        clean_audio(audio_file, output_file)

# Основная функция для запуска обработки
def main():
    # Пример аудиофайлов для обработки (замените на ваши пути)
    audio_files = [
        '/Users/daniil/Хакатоны/ЦП СВФО/1/data/ржд 1/ESC_DATASET_v1.2/luga/02_11_2023/2023_11_02__10_32_31.wav',
        '/Users/daniil/Хакатоны/ЦП СВФО/1/data/ржд 1/ESC_DATASET_v1.2/luga/02_11_2023/2023_11_02__10_43_52.wav'
    ]
    
    # Директория для сохранения очищенных файлов
    output_dir = '/mnt/data/cleaned_audio'
    
    # Запуск обработки файлов
    process_all_files(audio_files, output_dir)

if __name__ == "__main__":
    main()

