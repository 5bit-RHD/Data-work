import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import noisereduce as nr
from scipy.signal import butter, filtfilt
import time  # Импортируем модуль времени
import librosa.display  # Убедимся, что librosa.display импортирован
import psutil  # Импортируем psutil для измерения использования памяти
import os


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # Переводим байты в МБ
    return mem


def bandpass_filter(data, lowcut, highcut, sample_rate, order=6):
    """
    Применяет полосовой фильтр Баттерворта к аудиосигналу.

    :param data: Входной аудиосигнал.
    :param lowcut: Нижняя граница частоты (Гц).
    :param highcut: Верхняя граница частоты (Гц).
    :param sample_rate: Частота дискретизации (Гц).
    :param order: Порядок фильтра.
    :return: Отфильтрованный аудиосигнал.
    """
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def process_audio_file(audio_file):
    mem_before = get_memory_usage()

    # Загрузка аудиофайла
    audio_data, sample_rate = librosa.load(audio_file, sr=None)

    # Измерение времени выполнения фильтрации и шумоподавления
    start_time = time.time()

    # Применение полосового фильтра
    LOWCUT = 300.0  # Нижняя граница частоты (Гц)
    HIGHCUT = 3400.0  # Верхняя граница частоты (Гц)
    filtered_audio = bandpass_filter(audio_data, LOWCUT, HIGHCUT, sample_rate, order=6)

    # Определение сегмента шума (например, первые 1 секунду)
    noise_duration = 1.0  # секунды
    noise_samples = int(noise_duration * sample_rate)
    noise_clip = filtered_audio[:noise_samples]

    # Применение шумоподавления с использованием профиля шума
    reduced_noise = nr.reduce_noise(y=filtered_audio,
                                    sr=sample_rate,
                                    y_noise=noise_clip,
                                    prop_decrease=1.0)

    end_time = time.time()
    mem_after = get_memory_usage()
    mem_used = mem_after - mem_before

    # Вывод информации о потреблении памяти и времени выполнения
    print(f"Файл: {audio_file}")
    print(f"Память, используемая шумоподавлением: {mem_used:.2f} МБ")
    noise_reduction_time = end_time - start_time
    print(f"Время выполнения шумоподавления: {noise_reduction_time:.4f} секунд")

    # Визуализация
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(filtered_audio, sr=sample_rate)
    plt.title(f'Аудиосигнал после фильтра (файл: {audio_file})')
    plt.xlabel('Время (сек)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(reduced_noise, sr=sample_rate)
    plt.title(f'Аудиосигнал после фильтра и шумоподавления (файл: {audio_file})')
    plt.xlabel('Время (сек)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()

    # Сохранение обработанного файла
    output_file = os.path.join('audios', 'processed', f'filtered_{os.path.basename(audio_file)}')
    sf.write(output_file, reduced_noise, sample_rate)
    print(f"Отфильтрованный аудиофайл сохранён как {output_file}")


# Убедимся, что папка для сохранения обработанных файлов существует
output_dir = os.path.join('audios', 'processed')
os.makedirs(output_dir, exist_ok=True)

# Получение списка всех файлов в папке "audios"
input_dir = 'audios'
for filename in os.listdir(input_dir):
    if filename.endswith('.wav'):
        audio_file_path = os.path.join(input_dir, filename)
        process_audio_file(audio_file_path)