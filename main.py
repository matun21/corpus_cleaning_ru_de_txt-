# src/main.py
import os
import sys
from datetime import datetime
from corpus_loader import ParallelCorpusLoader
from parallel_cleaner import ParallelCorpusCleaner

def main():
    """Основной пайплайн очистки корпуса"""
    
    # Конфигурация
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Пути к исходным данным
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    RU_RAW_FILE = os.path.join(RAW_DIR, "Tatoeba.ru-de.ru")
    DE_RAW_FILE = os.path.join(RAW_DIR, "Tatoeba.ru-de.de")
    
    # Пути для сохранения результатов
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    
    # Выходные файлы
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RU_CLEANED_FILE = os.path.join(CLEANED_DIR, f"tatoeba_cleaned_ru_{timestamp}.txt")
    DE_CLEANED_FILE = os.path.join(CLEANED_DIR, f"tatoeba_cleaned_de_{timestamp}.txt")
    LOG_FILE = os.path.join(LOGS_DIR, f"cleaning_log_{timestamp}.txt")
    DETAILED_LOG_FILE = os.path.join(LOGS_DIR, f"detailed_log_{timestamp}.txt")
    STATS_FILE = os.path.join(RESULTS_DIR, f"cleaning_stats_{timestamp}.txt")
    
    # Создание директорий
    os.makedirs(CLEANED_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 70)
    print("ПАРАЛЛЕЛЬНАЯ ОЧИСТКА КОРПУСА: РУССКИЙ-НЕМЕЦКИЙ")
    print("=" * 70)
    print(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Директория проекта: {BASE_DIR}")
    print()
    
    # Шаг 1: Загрузка корпуса
    print("[Шаг 1] Загрузка параллельного корпуса...")
    try:
        ru_sentences, de_sentences = ParallelCorpusLoader.load_corpus(
            src_file=RU_RAW_FILE,
            tgt_file=DE_RAW_FILE,
            max_lines=None  # Загрузить все
        )
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print("Пожалуйста, убедитесь что:")
        print(f"  1. Файлы находятся в директории: {RAW_DIR}")
        print(f"  2. Имена файлов: Tatoeba.ru-de.ru и Tatoeba.ru-de.de")
        print("\nВы можете скачать корпус командой:")
        print("wget https://object.pouta.csc.fi/OPUS-Tatoeba/v2021-07-22/moses/ru-de.txt.zip")
        print("unzip ru-de.txt.zip")
        sys.exit(1)
    
    # Шаг 2: Инициализация очистителя
    print("\n[Шаг 2] Инициализация очистителя...")
    cleaner = ParallelCorpusCleaner(ru_sentences, de_sentences)
    
    # Шаг 3: Запуск пайплайна очистки
    print("\n[Шаг 3] Запуск пайплайна очистки...")
    cleaned_ru, cleaned_de = cleaner.run_pipeline(
        min_words=3,        # Минимум 3 слова
        max_ratio=2.0,      # Максимальное отношение длин 2.0
        min_alpha=0.8,      # Минимум 80% алфавитных символов
        ru_threshold=0.7,   # Порог уверенности для русского
        de_threshold=0.7    # Порог уверенности для немецкого
    )
    
    # Шаг 4: Сохранение результатов
    print("\n[Шаг 4] Сохранение результатов...")
    
    # Сохранение очищенного корпуса
    ParallelCorpusLoader.save_corpus(
        src_sentences=cleaned_ru,
        tgt_sentences=cleaned_de,
        src_output=RU_CLEANED_FILE,
        tgt_output=DE_CLEANED_FILE
    )
    
    # Сохранение логов
    cleaner.save_logs(LOG_FILE)
    cleaner.save_detailed_logs(DETAILED_LOG_FILE)
    
    # Сохранение статистики
    cleaner.save_statistics(STATS_FILE)
    
    # Шаг 5: Итоговый отчет
    print("\n" + "=" * 70)
    print("ОЧИСТКА КОРПУСА ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 70)
    print("\nСозданные файлы:")
    print(f"  1. Очищенный русский корпус: {RU_CLEANED_FILE}")
    print(f"  2. Очищенный немецкий корпус: {DE_CLEANED_FILE}")
    print(f"  3. Лог очистки: {LOG_FILE}")
    print(f"  4. Детальный лог: {DETAILED_LOG_FILE}")
    print(f"  5. Статистика: {STATS_FILE}")
    
    # Показ итоговой статистики
    print("\nИтоговая статистика:")
    cleaner.print_statistics()
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
