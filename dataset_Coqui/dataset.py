import json
import os
import csv
import librosa
import soundfile as sf
import time
import psutil
from sklearn.model_selection import train_test_split
import shutil

# Функция для конвертации MP3 или других файлов в WAV формат (16 кГц, моно)
def convert_to_wav(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith('.mp3') or file_name.endswith('.wav'):
                file_path = os.path.join(root, file_name)
                output_path = os.path.join(output_dir, file_name.replace('.mp3', '.wav'))

                try:
                    # Загружаем аудиофайл с помощью librosa
                    audio, sr = librosa.load(file_path, sr=16000, mono=True)  # Приводим к 16 кГц и моно
                    sf.write(output_path, audio, 16000)  # Сохраняем файл в WAV формате
                    print(f"Конвертирован {file_path} в {output_path}")
                except Exception as e:
                    print(f"Ошибка при конвертации файла {file_path}: {e}")

# Функция для создания CSV файла из JSON аннотаций, исключая шумы
def create_csv_from_json(json_path, audio_base_dir, csv_file, noise_dir=None):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(csv_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        for entry in data:
            audio_file = entry['audio_filepath']
            transcription = entry['text']

            # Полный путь к аудиофайлу
            audio_path = os.path.join(audio_base_dir, audio_file)
            
            # Пропускаем файлы с шумом, если noise_dir указана
            if noise_dir and noise_dir in audio_path:
                print(f"Пропуск файла с шумом: {audio_path}")
                continue

            # Проверяем наличие аудиофайла
            if os.path.exists(audio_path):
                writer.writerow([audio_path, transcription])
            else:
                print(f"Файл не найден: {audio_path}")

# Функция для обработки папки luga с учетом структуры
def process_luga(luga_dir, output_dir, noise_dir):
    # Обрабатываем каждую датированную папку
    for folder in os.listdir(luga_dir):
        folder_path = os.path.join(luga_dir, folder)
        
        # Пропускаем папку с шумами
        if folder == "noise":
            continue
        
        output_folder = os.path.join(output_dir, folder)
        print(f"Конвертация аудио для папки {folder}")
        convert_to_wav(folder_path, output_folder)

# Разделение данных на обучающую, валидационную и тестовую выборки
def split_data(csv_file, train_csv, dev_csv, test_csv, test_size=0.2, dev_size=0.1):
    with open(csv_file, 'r', encoding='utf-8') as f:
        data = [row for row in csv.reader(f)]
    
    train_data, test_data = train_test_split(data, test_size=test_size)
    train_data, dev_data = train_test_split(train_data, test_size=dev_size)
    
    # Сохраняем результаты
    with open(train_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(train_data)
    
    with open(dev_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(dev_data)
    
    with open(test_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(test_data)

# Метрика времени выполнения и использования памяти
def get_performance_metrics():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # В МБ
    cpu_time = time.process_time()  # Время CPU
    return mem, cpu_time

# Пути к папкам с аннотациями и аудиофайлами
annotations_dir = "/Users/daniil/Хакатоны/ЦП СВФО/Data-work/data/ржд 1/ESC_DATASET_v1.2/annotation"
audio_dirs = {
    'hr_bot_clear.json': "/Users/daniil/Хакатоны/ЦП СВФО/Data-work/data/ржд 1/ESC_DATASET_v1.2/hr_bot_clearr",
    'hr_bot_noise.json': "/Users/daniil/Хакатоны/ЦП СВФО/Data-work/data/ржд 1/ESC_DATASET_v1.2/hr_bot_noise",
    'hr_bot_synt.json': "/Users/daniil/Хакатоны/ЦП СВФО/Data-work/data/ржд 1/ESC_DATASET_v1.2/hr_bot_synt",
    'luga.json': "/Users/daniil/Хакатоны/ЦП СВФО/Data-work/data/ржд 1/ESC_DATASET_v1.2/luga"
}

# Папка с шумами
noise_dir = "/path/to/ESC_DATASET_v1.2/luga/noise"

# Папка для сохранения конвертированных файлов
output_dirs = {
    'hr_bot_clear.json': "/Users/daniil/Хакатоны/ЦП СВФО/Data-work/dataset_Coqui/dataset/hr_bot_clear_converted",
    'hr_bot_noise.json': "/Users/daniil/Хакатоны/ЦП СВФО/Data-work/dataset_Coqui/dataset/hr_bot_noise_converted",
    'hr_bot_synt.json': "/Users/daniil/Хакатоны/ЦП СВФО/Data-work/dataset_Coqui/dataset/hr_bot_synt_converted",
    'luga.json': "/Users/daniil/Хакатоны/ЦП СВФО/Data-work/dataset_Coqui/dataset/luga_converted"
}

# Создаем CSV файлы и конвертируем аудиофайлы
for json_file in audio_dirs.keys():
    json_path = os.path.join(annotations_dir, json_file)
    audio_dir = audio_dirs[json_file]
    output_dir = output_dirs[json_file]
    output_csv = json_file.replace('.json', '.csv')

    # Конвертируем файлы в WAV
    if json_file == 'luga.json':
        print(f"Обработка папки luga")
        process_luga(audio_dir, output_dir, noise_dir)
    else:
        print(f"Конвертация аудио для {json_file}")
        convert_to_wav(audio_dir, output_dir)

    # Создаем CSV файлы, игнорируя шумы
    if json_file == 'luga.json':
        create_csv_from_json(json_path, output_dir, output_csv, noise_dir=noise_dir)
    else:
        create_csv_from_json(json_path, output_dir, output_csv)
    
    print(f"CSV файл для {json_file} создан: {output_csv}")

# Разделение на обучающую, валидационную и тестовую выборки
split_data('luga.csv', 'train.csv', 'dev.csv', 'test.csv')

# Печать метрик
memory, cpu_time = get_performance_metrics()
print(f"Использование памяти: {memory:.2f} МБ, Время CPU: {cpu_time:.2f} секунд")
