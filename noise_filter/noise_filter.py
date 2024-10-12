import librosa
import noisereduce as nr
import soundfile as sf
import time
import numpy as np

# Функция для разбивки аудио на фрагменты
def split_audio(y, sr, segment_duration=0.05):
    segment_samples = int(sr * segment_duration)  # Количество сэмплов в сегменте
    return [y[i:i + segment_samples] for i in range(0, len(y), segment_samples)]

# Функция для вычисления уровня шума
def compute_noise_level(y_segment, threshold=0.005):
    rms = np.sqrt(np.mean(np.square(y_segment)))
    return rms < threshold

# Основная функция обработки
def clean_audio_file(audio_file, output_file, segment_duration=0.05, noise_threshold=0.005):
    start_time = time.time()

    # Загрузка аудиофайла
    y, sr = librosa.load(audio_file, sr=None)

    # Разбиваем аудио на сегменты
    segments = split_audio(y, sr, segment_duration)

    # Обрабатываем только шумные сегменты
    cleaned_segments = []
    for segment in segments:
        if compute_noise_level(segment, noise_threshold):
            # Если сегмент шумный, применяем шумоподавление
            cleaned_segment = nr.reduce_noise(y=segment, sr=sr, prop_decrease=0.9)
        else:
            # Чистые сегменты оставляем без изменений
            cleaned_segment = segment
        cleaned_segments.append(cleaned_segment)

    # Соединяем обработанные сегменты обратно
    cleaned_audio = np.concatenate(cleaned_segments)

    # Сохранение очищенного аудиофайла
    sf.write(output_file, cleaned_audio, sr)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

# Запуск
audio_file = '/Users/daniil/Хакатоны/ЦП СВФО/1/data/ржд 1/ESC_DATASET_v1.2/luga/15_11_2023/2023_11_15__09_24_51.wav'
output_file = 'cleaned_1.wav'
clean_audio_file(audio_file, output_file)
