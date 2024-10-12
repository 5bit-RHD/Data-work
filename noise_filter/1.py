import os
import librosa
import soundfile as sf
import noisereduce as nr
import time
import psutil
from concurrent.futures import ThreadPoolExecutor

# Функция для очистки аудиофайла с использованием noisereduce
def clean_audio(audio_file, output_file, sr=None, noise_threshold=0.005):
    try:
        start_time = time.time()  # Замер времени начала
        process = psutil.Process(os.getpid())  # Текущий процесс для замеров памяти
        y, sr = librosa.load(audio_file, sr=sr)
        cleaned_data = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.9, time_mask_smooth_ms=200)
        sf.write(output_file, cleaned_data, sr)
        end_time = time.time()
        elapsed_time = end_time - start_time
        memory_used = process.memory_info().rss / (1024 * 1024)  # в МБ
        print(f"Processed {audio_file} in {elapsed_time:.2f} seconds, Memory used: {memory_used:.2f} MB")
    
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")

# Функция для многопоточной обработки всех файлов
def process_all_files(audio_files, output_dir, max_workers=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Используем ThreadPoolExecutor для многопоточной обработки
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            output_file = os.path.join(output_dir, 'cleaned_' + filename)
            futures.append(executor.submit(clean_audio, audio_file, output_file))
        
        # Ожидаем завершения всех потоков
        for future in futures:
            future.result()  # Получаем результат обработки каждого файла

# Основная функция для запуска обработки
def main():
    audio_files = [
        '/Users/daniil/Хакатоны/ЦП СВФО/1/data/ржд 1/ESC_DATASET_v1.2/luga/02_11_2023/2023_11_02__10_33_18.wav',
        '/Users/daniil/Хакатоны/ЦП СВФО/1/data/ржд 1/ESC_DATASET_v1.2/luga/02_11_2023/2023_11_02__10_34_29.wav'
    ]
    
    output_dir = '/Users/daniil/Хакатоны/ЦП СВФО/1/noise_filter/cleaned_audio'
    
    process_all_files(audio_files, output_dir, max_workers=6)

if __name__ == "__main__":
    main()
