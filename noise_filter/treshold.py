import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import noisereduce as nr
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d
import time
import librosa.display
import psutil
import os
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

# Функция для вычисления огибающей сигнала
def calculate_envelope(signal, frame_size=1024, smooth_factor=100):
    envelope = np.abs(signal)
    smoothed_envelope = uniform_filter1d(envelope, size=smooth_factor)
    return smoothed_envelope

# Функция для обрезки аудиосигнала по огибающей
def trim_audio_by_envelope(audio, envelope, threshold=0.02):
    normalized_envelope = envelope / np.max(envelope)
    speech_indices = np.where(normalized_envelope > threshold)[0]
    if len(speech_indices) > 0:
        start_idx = speech_indices[0]
        end_idx = speech_indices[-1]
        return audio[start_idx:end_idx], start_idx, end_idx
    else:
        return audio, 0, len(audio)

# Функция для конвертации MP3 в WAV
def convert_mp3_to_wav(mp3_file, output_dir):
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = os.path.join(output_dir, os.path.splitext(os.path.basename(mp3_file))[0] + '.wav')
    audio.export(wav_file, format="wav")
    return wav_file

# Основная функция обработки аудиофайла
def process_audio_file(audio_file, output_dir, lowcut=300.0, highcut=3400.0, threshold=0.2):
    # Загрузка аудиофайла
    audio_data, sample_rate = librosa.load(audio_file, sr=None)

    # Применение полосового фильтра
    filtered_audio = bandpass_filter(audio_data, lowcut, highcut, sample_rate, order=6)

    # Вычисление огибающей сигнала
    envelope = calculate_envelope(filtered_audio)

    # Сохранение графика огибающей
    plt.figure(figsize=(14, 4))
    plt.plot(envelope)
    plt.title('Огибающая сигнала')
    plt.xlabel('Фреймы')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    envelope_image = os.path.join(output_dir, f'{os.path.basename(audio_file)}_envelope.png')
    plt.savefig(envelope_image)
    plt.close()

    # Обрезка аудиосигнала по огибающей
    trimmed_audio, start_idx, end_idx = trim_audio_by_envelope(filtered_audio, envelope, threshold=threshold)

    # Сохранение графика обрезанного аудиосигнала
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(trimmed_audio, sr=sample_rate)
    plt.title('Обрезанный аудиосигнал (по огибающей)')
    plt.xlabel('Время (сек)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    trimmed_image = os.path.join(output_dir, f'{os.path.basename(audio_file)}_trimmed.png')
    plt.savefig(trimmed_image)
    plt.close()

    # Применение шумоподавления
    noise_duration = 1.0  # секунды
    noise_samples = int(noise_duration * sample_rate)
    noise_clip = trimmed_audio[:noise_samples]

    # Замер времени шумоподавления
    start_time = time.time()
    reduced_noise = nr.reduce_noise(y=trimmed_audio, sr=sample_rate, y_noise=noise_clip, prop_decrease=1.0)
    end_time = time.time()
    noise_reduction_time = end_time - start_time

    # Сохранение графика финального аудиосигнала
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(reduced_noise, sr=sample_rate)
    plt.title('Аудиосигнал после шумоподавления')
    plt.xlabel('Время (сек)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    noise_reduction_image = os.path.join(output_dir, f'{os.path.basename(audio_file)}_final.png')
    plt.savefig(noise_reduction_image)
    plt.close()

    # Сохранение обработанного аудиофайла
    output_file = os.path.join(output_dir, f'trimmed_filtered_{os.path.basename(audio_file)}')
    sf.write(output_file, reduced_noise, sample_rate)

    return output_file, noise_reduction_time

# Функция для обработки всех файлов в папке
def process_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp3'):
                mp3_file = os.path.join(root, file)
                audio_file = convert_mp3_to_wav(mp3_file, output_dir)
            elif file.endswith('.wav'):
                audio_file = os.path.join(root, file)
            else:
                continue  # Пропускаем файлы других форматов

            mem_before = get_memory_usage()

            # Обработка аудиофайла
            processed_file, noise_reduction_time = process_audio_file(audio_file, output_dir)

            # Измерение памяти после обработки
            mem_after = get_memory_usage()
            mem_used = mem_after - mem_before

            # Вывод информации
            print(f"Файл {processed_file} обработан. Время шумоподавления: {noise_reduction_time:.2f} секунд, Память: {mem_used:.2f} МБ")

# Основная функция
def main():
    input_dir = '/Users/daniil/Хакатоны/ЦП СВФО/data/ржд 1/ESC_DATASET_v1.2/luga/02_11_2023'  # Папка с аудиофайлами
    output_dir = '/Users/daniil/Хакатоны/ЦП СВФО/noise_filter/cleaned_aud_treshold'  # Папка для сохранения очищенных аудиофайлов

    # Запуск обработки всех файлов в папке
    process_folder(input_dir, output_dir)

if __name__ == "__main__":
    main()
