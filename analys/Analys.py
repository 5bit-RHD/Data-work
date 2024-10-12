import os
import json
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Проверка наличия аудиофайлов и генерация отчета
def check_audio_files(json_file, audio_dir):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    report = []
    total_duration = 0
    durations = []
    empty_texts = []
    missing_files = []
    
    for entry in data:
        audio_path = os.path.join(audio_dir, entry['audio_filepath'])
        row = {
            'audio_filepath': entry['audio_filepath'],
            'text': entry['text'],
            'label': entry['label'],
            'attribute': entry.get('attribute', None),
            'exists': os.path.exists(audio_path),
            'duration': None
        }
        
        if not os.path.exists(audio_path):
            missing_files.append(audio_path)
        else:
            # Рассчет продолжительности аудиофайлов
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            durations.append(duration)
            row['duration'] = duration
            total_duration += duration
        
        if not entry['text'].strip():
            empty_texts.append(entry['id'])
        
        report.append(row)

    # Сохранение отчета в CSV
    df_report = pd.DataFrame(report)
    df_report.to_csv(f'{json_file}_audio_report.csv', index=False)
    
    print(f'Total duration of audio in {json_file}: {total_duration:.2f} seconds')
    print(f'Missing files: {len(missing_files)}')
    print(f'Empty texts: {len(empty_texts)}')

    return df_report, durations

# Анализ метаданных JSON и визуализация
def analyze_json_metadata(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    classes = [entry['label'] for entry in data]
    attributes = [entry['attribute'] for entry in data if 'attribute' in entry]
    texts = [entry['text'] for entry in data]
    
    # Визуализация распределения классов
    unique_classes, class_counts = np.unique(classes, return_counts=True)
    class_dist = pd.DataFrame({'Class': unique_classes, 'Count': class_counts})
    

# Визуализация распределения классов
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Class', y='Count', data=class_dist, palette='viridis', hue=None, legend=False)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{json_file}_class_distribution.png')
    plt.show()

    # Визуализация распределения атрибутов (если есть)
    if attributes:
        unique_attributes, attr_counts = np.unique(attributes, return_counts=True)
        attr_dist = pd.DataFrame({'Attribute': unique_attributes, 'Count': attr_counts})
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Attribute', y='Count', data=attr_dist, palette='magma', hue=None, legend=False)
        plt.title('Attribute Distribution')
        plt.xlabel('Attribute')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{json_file}_attribute_distribution.png')
        plt.show()


    
    # Средняя длина текста команд
    text_lengths = [len(t.split()) for t in texts]
    avg_text_length = np.mean(text_lengths)
    print(f'Average text length in {json_file}: {avg_text_length:.2f} words')

    # Сохранение данных о классах и атрибутах
    class_dist.to_csv(f'{json_file}_class_report.csv', index=False)
    if attributes:
        attr_dist.to_csv(f'{json_file}_attribute_report.csv', index=False)

    return classes, text_lengths

# Построение и сохранение спектрограммы
def plot_spectrogram(audio_file, save_path):
    y, sr = librosa.load(audio_file)
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram for {os.path.basename(audio_file)}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Основной процесс анализа каждого набора данных
def analyze_dataset(json_file, audio_dir, analyze_spectrogram=False):
    print(f'Analyzing dataset: {json_file}')
    
    # 1. Проверка аудиофайлов и расчет их продолжительности
    df_report, durations = check_audio_files(json_file, audio_dir)
    
    # 2. Анализ метаданных JSON
    classes, text_lengths = analyze_json_metadata(json_file)
    
    # Визуализация распределений
    plt.figure(figsize=(10, 5))
    
    # Распределение классов
    plt.subplot(1, 2, 1)
    plt.hist(classes, bins=len(set(classes)), color='blue', alpha=0.7)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Распределение продолжительности аудиофайлов
    plt.subplot(1, 2, 2)
    plt.hist(durations, bins=20, color='green', alpha=0.7)
    plt.title('Audio Duration Distribution')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'{json_file}_distribution.png')
    plt.show()
    
    # Построение спектрограмм для первых нескольких аудиофайлов
    if analyze_spectrogram:
        for entry in df_report[df_report['exists']].head(5).itertuples():
            audio_path = os.path.join(audio_dir, entry.audio_filepath)
            save_path = f'spectrogram_{os.path.basename(audio_path)}.png'
            plot_spectrogram(audio_path, save_path)

# Основной процесс для всех наборов данных
def main():
    base_dir = '/Users/daniil/Хакатоны/ЦП СВФО/1/data/ржд 1/ESC_DATASET_v1.2'
    
    datasets = {
        'hr_bot_clear': 'annotation/hr_bot_clear.json',
        'hr_bot_noise': 'annotation/hr_bot_noise.json',
        'hr_bot_synt': 'annotation/hr_bot_synt.json',
        'luga': 'annotation/luga.json'
    }
    
    for dataset_name, json_file in datasets.items():
        print(f'Processing dataset: {dataset_name}')
        audio_dir = os.path.join(base_dir, dataset_name)
        json_path = os.path.join(base_dir, json_file)
        analyze_dataset(json_path, audio_dir, analyze_spectrogram=(dataset_name == 'hr_bot_noise'))

if __name__ == "__main__":
    main()
