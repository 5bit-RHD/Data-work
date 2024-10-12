import os
from pydub import AudioSegment

# Функция для конвертации MP3 в WAV
def convert_mp3_to_wav(input_dir, output_dir):
    # Проверяем, существует ли выходная папка, и создаем, если нужно
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Проход по всем файлам в указанной папке
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mp3"):
                mp3_file = os.path.join(root, file)
                wav_file = os.path.join(output_dir, os.path.splitext(file)[0] + '.wav')

                try:
                    # Конвертация MP3 в WAV
                    audio = AudioSegment.from_mp3(mp3_file)
                    audio.export(wav_file, format="wav")

                    print(f"Конвертирован {file} в {wav_file}")

                except Exception as e:
                    print(f"Ошибка при конвертации файла {file}: {e}")

def main():
    input_dir = '/Users/daniil/Хакатоны/ЦП СВФО/data/ржд 1/ESC_DATASET_v1.2/hr_bot_clear'  # Папка с MP3 файлами
    output_dir = '/Users/daniil/Хакатоны/ЦП СВФО/noise_filter/hr_bot_clear'  # Папка для сохранения WAV файлов
    convert_mp3_to_wav(input_dir, output_dir)

if __name__ == "__main__":
    main()




