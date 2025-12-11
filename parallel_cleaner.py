# src/parallel_cleaner.py
import os
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import logging

@dataclass
class CleaningStats:
    """Статистика очистки"""
    initial_pairs: int = 0
    removed_by_length: int = 0
    removed_by_ratio: int = 0
    removed_by_alpha: int = 0
    removed_by_language: int = 0
    final_pairs: int = 0
    total_removed: int = 0
    
    def update_final(self):
        """Обновляет итоговые значения"""
        self.total_removed = (self.removed_by_length + self.removed_by_ratio + 
                            self.removed_by_alpha + self.removed_by_language)
        self.final_pairs = self.initial_pairs - self.total_removed

class ParallelCorpusCleaner:
    """Основной класс для очистки параллельного корпуса"""
    
    def __init__(self, ru_sentences: List[str], de_sentences: List[str]):
        """
        Инициализация очистителя
        
        Args:
            ru_sentences: список русских предложений
            de_sentences: список немецких предложений
        """
        # Проверка входных данных
        if len(ru_sentences) != len(de_sentences):
            raise ValueError(f"Количество строк не совпадает: "
                           f"ru={len(ru_sentences)}, de={len(de_sentences)}")
        
        self.ru_original = ru_sentences
        self.de_original = de_sentences
        self.ru_current = ru_sentences.copy()
        self.de_current = de_sentences.copy()
        
        # Логи и статистика
        self.log_entries: List[Dict[str, Any]] = []
        self.stats = CleaningStats(initial_pairs=len(ru_sentences))
        
        # Инициализация фильтров
        self._init_filters()
        
        # Инициализация детектора языка
        self._init_language_detector()
        
        print(f"Инициализирован очиститель с {self.stats.initial_pairs} парами")
    
    def _init_filters(self):
        """Инициализация набора символов для фильтров"""
        # Русские символы (кириллица)
        self.cyrillic_chars = self._get_cyrillic_chars()
        
        # Немецкие символы (латиница + умлауты)
        self.german_chars = self._get_german_chars()
    
    def _init_language_detector(self):
        """Инициализация детектора языка"""
        try:
            from fasttext_langdetect import detect
            self.detect = detect
            self.langdetect_available = True
            print("Детектор языка fasttext инициализирован")
        except ImportError as e:
            print(f"Внимание: fasttext-langdetect не установлен ({e})")
            print("Фильтрация по языку будет пропущена")
            self.langdetect_available = False
    
    def _get_cyrillic_chars(self) -> set:
        """Возвращает множество кириллических символов"""
        cyr_chars = set()
        # Основной кириллический диапазон
        for code in range(1040, 1104):  # А-я
            cyr_chars.add(chr(code))
        # Ё и ё
        cyr_chars.add('Ё')
        cyr_chars.add('ё')
        # Добавляем возможные дополнительные символы
        cyr_chars.update(['№', '°'])
        return cyr_chars
    
    def _get_german_chars(self) -> set:
        """Возвращает множество немецких символов"""
        de_chars = set()
        # Латинские буквы A-Z, a-z
        for code in range(65, 91):  # A-Z
            de_chars.add(chr(code))
        for code in range(97, 123):  # a-z
            de_chars.add(chr(code))
        # Немецкие умлауты и эсцет
        de_chars.update(['ä', 'ö', 'ü', 'ß', 'Ä', 'Ö', 'Ü'])
        return de_chars
    
    def clean_short_sentences(self, min_words: int = 3) -> Tuple[List[str], List[str]]:
        """
        Фильтр 1: Удаление слишком коротких предложений
        
        Args:
            min_words: минимальное количество слов в предложении
        
        Returns:
            Отфильтрованные списки предложений
        """
        print(f"\n[1] Фильтрация коротких предложений (< {min_words} слов)...")
        
        filtered_ru, filtered_de = [], []
        
        for ru, de in tqdm(zip(self.ru_current, self.de_current), 
                          total=len(self.ru_current),
                          desc="Обработка"):
            # Разделение на слова (простейший способ)
            ru_words = ru.split()
            de_words = de.split()
            
            # Проверка минимальной длины
            if len(ru_words) >= min_words and len(de_words) >= min_words:
                filtered_ru.append(ru)
                filtered_de.append(de)
            else:
                self._log_removal(
                    filter_type="SHORT_LENGTH",
                    ru_sentence=ru,
                    de_sentence=de,
                    info=f"ru_words: {len(ru_words)}, de_words: {len(de_words)}"
                )
                self.stats.removed_by_length += 1
        
        self.ru_current, self.de_current = filtered_ru, filtered_de
        print(f"  Сохранено: {len(filtered_ru)} пар")
        print(f"  Удалено: {self.stats.removed_by_length} пар")
        
        return filtered_ru, filtered_de
    
    def clean_length_ratio(self, max_ratio: float = 2.0) -> Tuple[List[str], List[str]]:
        """
        Фильтр 2: Удаление пар с большой разницей в длине
        
        Args:
            max_ratio: максимальное допустимое отношение длин
        
        Returns:
            Отфильтрованные списки предложений
        """
        print(f"\n[2] Фильтрация по разнице длины (макс. отношение: {max_ratio})...")
        
        filtered_ru, filtered_de = [], []
        
        for ru, de in tqdm(zip(self.ru_current, self.de_current),
                          total=len(self.ru_current),
                          desc="Обработка"):
            # Вычисление длин в символах
            ru_len = len(ru)
            de_len = len(de)
            
            # Вычисление отношения длин
            if min(ru_len, de_len) == 0:
                ratio = float('inf')
            else:
                ratio = max(ru_len, de_len) / min(ru_len, de_len)
            
            # Проверка отношения
            if ratio <= max_ratio:
                filtered_ru.append(ru)
                filtered_de.append(de)
            else:
                self._log_removal(
                    filter_type="LENGTH_RATIO",
                    ru_sentence=ru,
                    de_sentence=de,
                    info=f"отношение: {ratio:.2f}, ru_len: {ru_len}, de_len: {de_len}"
                )
                self.stats.removed_by_ratio += 1
        
        self.ru_current, self.de_current = filtered_ru, filtered_de
        print(f"  Сохранено: {len(filtered_ru)} пар")
        print(f"  Удалено: {self.stats.removed_by_ratio} пар")
        
        return filtered_ru, filtered_de
    
    def clean_alpha_ratio(self, min_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
        """
        Фильтр 3: Удаление пар с низкой долей алфавитных символов
        
        Args:
            min_ratio: минимальная доля алфавитных символов
        
        Returns:
            Отфильтрованные списки предложений
        """
        print(f"\n[3] Фильтрация по алфавитным символам (мин. доля: {min_ratio})...")
        
        filtered_ru, filtered_de = [], []
        
        for ru, de in tqdm(zip(self.ru_current, self.de_current),
                          total=len(self.ru_current),
                          desc="Обработка"):
            # Вычисление доли алфавитных символов для русского
            if ru:
                ru_alpha = sum(1 for char in ru if char in self.cyrillic_chars)
                ru_ratio = ru_alpha / len(ru)
            else:
                ru_ratio = 0
            
            # Вычисление доли алфавитных символов для немецкого
            if de:
                de_alpha = sum(1 for char in de if char in self.german_chars)
                de_ratio = de_alpha / len(de)
            else:
                de_ratio = 0
            
            # Проверка минимальной доли
            if ru_ratio >= min_ratio and de_ratio >= min_ratio:
                filtered_ru.append(ru)
                filtered_de.append(de)
            else:
                self._log_removal(
                    filter_type="LOW_ALPHA",
                    ru_sentence=ru,
                    de_sentence=de,
                    info=f"ru_alpha: {ru_ratio:.3f}, de_alpha: {de_ratio:.3f}"
                )
                self.stats.removed_by_alpha += 1
        
        self.ru_current, self.de_current = filtered_ru, filtered_de
        print(f"  Сохранено: {len(filtered_ru)} пар")
        print(f"  Удалено: {self.stats.removed_by_alpha} пар")
        
        return filtered_ru, filtered_de
    
    def clean_by_language(self, 
                         ru_threshold: float = 0.7, 
                         de_threshold: float = 0.7) -> Tuple[List[str], List[str]]:
        """
        Фильтр 4: Удаление пар с неправильно определенным языком
        
        Args:
            ru_threshold: порог уверенности для русского
            de_threshold: порог уверенности для немецкого
        
        Returns:
            Отфильтрованные списки предложений
        """
        if not self.langdetect_available:
            print("\n[4] Пропуск фильтрации по языку (fasttext не доступен)")
            return self.ru_current, self.de_current
        
        print(f"\n[4] Фильтрация по языку (пороги: ru≥{ru_threshold}, de≥{de_threshold})...")
        
        filtered_ru, filtered_de = [], []
        
        for ru, de in tqdm(zip(self.ru_current, self.de_current),
                          total=len(self.ru_current),
                          desc="Определение языка"):
            try:
                # Детекция языка для русского текста
                ru_detection = self.detect(ru)
                ru_is_valid = (ru_detection['lang'] == 'ru' and 
                             ru_detection['prob'] >= ru_threshold)
                
                # Детекция языка для немецкого текста
                de_detection = self.detect(de)
                de_is_valid = (de_detection['lang'] == 'de' and 
                             de_detection['prob'] >= de_threshold)
                
                # Проверка обоих языков
                if ru_is_valid and de_is_valid:
                    filtered_ru.append(ru)
                    filtered_de.append(de)
                else:
                    self._log_removal(
                        filter_type="WRONG_LANGUAGE",
                        ru_sentence=ru,
                        de_sentence=de,
                        info=(f"ru: {ru_detection['lang']}({ru_detection['prob']:.3f}), "
                            f"de: {de_detection['lang']}({de_detection['prob']:.3f})")
                    )
                    self.stats.removed_by_language += 1
                    
            except Exception as e:
                # В случае ошибки оставляем пару, но логируем
                filtered_ru.append(ru)
                filtered_de.append(de)
                self._log_removal(
                    filter_type="LANG_DETECT_ERROR",
                    ru_sentence=ru,
                    de_sentence=de,
                    info=f"Ошибка: {str(e)}"
                )
                # Не увеличиваем счетчик удаленных, так как сохранили
        
        self.ru_current, self.de_current = filtered_ru, filtered_de
        print(f"  Сохранено: {len(filtered_ru)} пар")
        print(f"  Удалено: {self.stats.removed_by_language} пар")
        
        return filtered_ru, filtered_de
    
    def _log_removal(self, filter_type: str, ru_sentence: str, 
                    de_sentence: str, info: str = ""):
        """
        Добавляет запись в лог об удалении пары
        
        Args:
            filter_type: тип фильтра
            ru_sentence: русское предложение
            de_sentence: немецкое предложение
            info: дополнительная информация
        """
        self.log_entries.append({
            'filter_type': filter_type,
            'ru_sentence': ru_sentence,
            'de_sentence': de_sentence,
            'info': info
        })
    
    def save_logs(self, log_file: str):
        """
        Сохраняет лог в указанном формате
        
        Args:
            log_file: путь к файлу лога
        """
        print(f"\nСохранение лога в {log_file}...")
        
        # Создание директории если нужно
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            for entry in self.log_entries:
                f.write(f"{entry['filter_type']}\n")
                f.write(f"{entry['ru_sentence']}\n")
                f.write(f"{entry['de_sentence']}\n")
                # Пустая строка между записями (необязательно, но удобно)
                # f.write("\n")
        
        print(f"  Сохранено {len(self.log_entries)} записей в лог")
    
    def save_detailed_logs(self, log_file: str):
        """
        Сохраняет детальный лог с дополнительной информацией
        
        Args:
            log_file: путь к файлу лога
        """
        print(f"\nСохранение детального лога в {log_file}...")
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("ДЕТАЛЬНЫЙ ЛОГ ОЧИСТКИ КОРПУСА\n")
            f.write("=" * 50 + "\n\n")
            
            for i, entry in enumerate(self.log_entries, 1):
                f.write(f"Запись #{i}\n")
                f.write(f"Тип фильтра: {entry['filter_type']}\n")
                f.write(f"Русское предложение: {entry['ru_sentence']}\n")
                f.write(f"Немецкое предложение: {entry['de_sentence']}\n")
                if entry['info']:
                    f.write(f"Доп. информация: {entry['info']}\n")
                f.write("-" * 50 + "\n\n")
    
    def save_statistics(self, stats_file: str):
        """
        Сохраняет статистику очистки
        
        Args:
            stats_file: путь к файлу статистики
        """
        print(f"\nСохранение статистики в {stats_file}...")
        
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        
        # Обновление итоговой статистики
        self.stats.update_final()
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("СТАТИСТИКА ОЧИСТКИ ПАРАЛЛЕЛЬНОГО КОРПУСА\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Исходное количество пар: {self.stats.initial_pairs:,}\n")
            f.write(f"Удалено по короткой длине: {self.stats.removed_by_length:,}\n")
            f.write(f"Удалено по разнице длины: {self.stats.removed_by_ratio:,}\n")
            f.write(f"Удалено по алфавитным символам: {self.stats.removed_by_alpha:,}\n")
            f.write(f"Удалено по языку: {self.stats.removed_by_language:,}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Всего удалено: {self.stats.total_removed:,}\n")
            f.write(f"Конечное количество пар: {self.stats.final_pairs:,}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Процент сохраненных: "
                  f"{(self.stats.final_pairs / self.stats.initial_pairs * 100):.2f}%\n")
            f.write(f"Процент удаленных: "
                  f"{(self.stats.total_removed / self.stats.initial_pairs * 100):.2f}%\n")
        
        print("Статистика сохранена")
    
    def print_statistics(self):
        """Выводит статистику очистки в консоль"""
        print("\n" + "=" * 60)
        print("СТАТИСТИКА ОЧИСТКИ ПАРАЛЛЕЛЬНОГО КОРПУСА")
        print("=" * 60)
        
        # Обновление итоговой статистики
        self.stats.update_final()
        
        print(f"Исходное количество пар: {self.stats.initial_pairs:,}")
        print(f"Удалено по короткой длине: {self.stats.removed_by_length:,}")
        print(f"Удалено по разнице длины: {self.stats.removed_by_ratio:,}")
        print(f"Удалено по алфавитным символам: {self.stats.removed_by_alpha:,}")
        print(f"Удалено по языку: {self.stats.removed_by_language:,}")
        print("-" * 60)
        print(f"Всего удалено: {self.stats.total_removed:,}")
        print(f"Конечное количество пар: {self.stats.final_pairs:,}")
        print("-" * 60)
        print(f"Процент сохраненных: "
              f"{(self.stats.final_pairs / self.stats.initial_pairs * 100):.2f}%")
        print(f"Процент удаленных: "
              f"{(self.stats.total_removed / self.stats.initial_pairs * 100):.2f}%")
        print("=" * 60)
    
    def run_pipeline(self, 
                    min_words: int = 3,
                    max_ratio: float = 2.0,
                    min_alpha: float = 0.8,
                    ru_threshold: float = 0.7,
                    de_threshold: float = 0.7) -> Tuple[List[str], List[str]]:
        """
        Запуск полного пайплайна очистки
        
        Args:
            min_words: минимальное количество слов
            max_ratio: максимальное отношение длин
            min_alpha: минимальная доля алфавитных символов
            ru_threshold: порог уверенности для русского
            de_threshold: порог уверенности для немецкого
        
        Returns:
            Очищенные списки предложений
        """
        print("=" * 60)
        print("ЗАПУСК ПАЙПЛАЙНА ОЧИСТКИ ПАРАЛЛЕЛЬНОГО КОРПУСА")
        print("=" * 60)
        
        # Применение фильтров в указанном порядке
        self.clean_short_sentences(min_words=min_words)
        self.clean_length_ratio(max_ratio=max_ratio)
        self.clean_alpha_ratio(min_ratio=min_alpha)
        self.clean_by_language(ru_threshold=ru_threshold, de_threshold=de_threshold)
        
        # Вывод итоговой статистики
        self.print_statistics()
        
        return self.ru_current, self.de_current
