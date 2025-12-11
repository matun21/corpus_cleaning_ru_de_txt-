# src/config.py
"""
Конфигурация параметров очистки корпуса
"""

class CleaningConfig:
    """Конфигурация параметров очистки"""
    
    # Параметры фильтров
    MIN_WORDS = 3           # Минимальное количество слов
    MAX_LENGTH_RATIO = 2.0  # Максимальное отношение длин
    MIN_ALPHA_RATIO = 0.8   # Минимальная доля алфавитных символов
    RU_LANG_THRESHOLD = 0.7 # Порог уверенности для русского
    DE_LANG_THRESHOLD = 0.7 # Порог уверенности для немецкого
    
    # Пути к данным
    BASE_DIR = "corpus_cleaning_ru_de_txt"
    RAW_DATA_DIR = "data/raw"
    CLEANED_DATA_DIR = "data/cleaned"
    LOGS_DIR = "logs"
    RESULTS_DIR = "results"
    
    # Имена файлов
    RU_RAW_FILE = "Tatoeba.ru-de.ru"
    DE_RAW_FILE = "Tatoeba.ru-de.de"
    
    # Кодировка файлов
    ENCODING = "utf-8"
    
    # Максимальное количество строк (для тестирования)
    MAX_LINES = None  # None - все строки
    
    # Включение/выключение фильтров
    ENABLE_LENGTH_FILTER = True
    ENABLE_RATIO_FILTER = True
    ENABLE_ALPHA_FILTER = True
    ENABLE_LANG_FILTER = True
