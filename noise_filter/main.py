import os
import librosa
import soundfile as sf
import noisereduce as nr
import psutil
import time
import numpy as np
from scipy.signal import butter, filtfilt
from pydub import AudioSegment

# Функция для измерения использования памяти
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # Переводим байты в МБ
    return mem

# Полосовой фильтр Баттерворта
def bandpass_filter(data, lowcut, highcut, sample_rate, order=6):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Функция для конвертации MP3 в WAV
def convert_mp3_to_wav(mp3_file, output_dir):
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = os.path.join(output_dir, os.path.splitext(os.path.basename(mp3_file))[0] + '.wav')
    audio.export(wav_file, format="wav")
    return wav_file

# Функция для обработки аудиофайла
def process_audio_file(audio_file, output_dir, lowcut=300.0, highcut=3400.0):
    # Загрузка аудиофайла
    audio_data, sample_rate = librosa.load(audio_file, sr=None)

    # Применение полосового фильтра
    filtered_audio = bandpass_filter(audio_data, lowcut, highcut, sample_rate, order=6)

    # Определение сегмента шума (например, первые 1 секунду)
    noise_duration = 1.0  # секунды
    noise_samples = int(noise_duration * sample_rate)
    noise_clip = filtered_audio[:noise_samples]

    # Применение шумоподавления с использованием профиля шума
    reduced_noise = nr.reduce_noise(y=filtered_audio, sr=sample_rate, y_noise=noise_clip, prop_decrease=1.0)

    # Сохранение обработанного файла
    output_file = os.path.join(output_dir, f'cleaned_{os.path.basename(audio_file)}')
    sf.write(output_file, reduced_noise, sample_rate)

    # Возвращаем информацию для логов
    return output_file

# Функция для обработки всех файлов в папке
def process_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Преобразуем MP3 в WAV, если нужно
            if file.endswith('.mp3'):
                mp3_file = os.path.join(root, file)
                audio_file = convert_mp3_to_wav(mp3_file, output_dir)
            elif file.endswith('.wav'):
                audio_file = os.path.join(root, file)
            else:
                continue  # Пропускаем файлы других форматов

            # Измерение памяти и времени до обработки
            mem_before = get_memory_usage()
            start_time = time.time()

            # Обработка аудиофайла
            processed_file = process_audio_file(audio_file, output_dir)

            # Измерение времени и памяти после обработки
            end_time = time.time()
            mem_after = get_memory_usage()
            mem_used = mem_after - mem_before
            execution_time = end_time - start_time

            # Вывод информации в консоль
            print(f"Файл {processed_file} обработан. Время: {execution_time:.2f} секунд, Память: {mem_used:.2f} МБ")

# Основная функция
def main():
    input_dir = '/Users/daniil/Хакатоны/ЦП СВФО/data/ржд 1/ESC_DATASET_v1.2/hr_bot_clear'  # Папка с аудиофайлами
    output_dir = '/Users/daniil/Хакатоны/ЦП СВФО/noise_filter/hr_bot_clear'  # Папка для сохранения очищенных аудиофайлов

    # Запуск обработки всех файлов в папке
    process_folder(input_dir, output_dir)

if __name__ == "__main__":
    main()

