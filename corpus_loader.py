# src/corpus_loader.py
import os
from typing import List, Tuple
from tqdm import tqdm

class ParallelCorpusLoader:
    """Загрузчик параллельного корпуса из текстовых файлов"""
    
    @staticmethod
    def load_corpus(src_file: str, tgt_file: str, 
                   max_lines: int = None,
                   encoding: str = 'utf-8') -> Tuple[List[str], List[str]]:
        """
        Загружает параллельный корпус из двух файлов
        
        Args:
            src_file: путь к файлу исходного языка (русский)
            tgt_file: путь к файлу целевого языка (немецкий)
            max_lines: максимальное количество строк для загрузки
            encoding: кодировка файлов
        
        Returns:
            Кортеж (src_sentences, tgt_sentences)
        """
        print(f"Загрузка корпуса:")
        print(f"  Русский: {src_file}")
        print(f"  Немецкий: {tgt_file}")
        
        # Проверка существования файлов
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"Файл не найден: {src_file}")
        if not os.path.exists(tgt_file):
            raise FileNotFoundError(f"Файл не найден: {tgt_file}")
        
        # Загрузка файлов
        with open(src_file, 'r', encoding=encoding) as f:
            src_lines = [line.strip() for line in f]
        
        with open(tgt_file, 'r', encoding=encoding) as f:
            tgt_lines = [line.strip() for line in f]
        
        # Проверка соответствия количества строк
        if len(src_lines) != len(tgt_lines):
            print(f"Внимание: количество строк не совпадает!")
            print(f"  Русский: {len(src_lines)} строк")
            print(f"  Немецкий: {len(tgt_lines)} строк")
            # Ограничиваем минимальным количеством
            min_len = min(len(src_lines), len(tgt_lines))
            src_lines = src_lines[:min_len]
            tgt_lines = tgt_lines[:min_len]
            print(f"  Ограничиваем до {min_len} пар")
        
        # Ограничение по максимальному количеству строк
        if max_lines and max_lines < len(src_lines):
            src_lines = src_lines[:max_lines]
            tgt_lines = tgt_lines[:max_lines]
            print(f"  Ограничено до {max_lines} пар")
        
        print(f"Загружено {len(src_lines)} параллельных пар")
        return src_lines, tgt_lines
    
    @staticmethod
    def save_corpus(src_sentences: List[str], tgt_sentences: List[str],
                   src_output: str, tgt_output: str,
                   encoding: str = 'utf-8'):
        """
        Сохраняет параллельный корпус в два файла
        
        Args:
            src_sentences: предложения на исходном языке
            tgt_sentences: предложения на целевом языке
            src_output: путь для сохранения исходного языка
            tgt_output: путь для сохранения целевого языка
            encoding: кодировка для сохранения
        """
        print(f"Сохранение корпуса:")
        print(f"  Русский: {src_output} ({len(src_sentences)} строк)")
        print(f"  Немецкий: {tgt_output} ({len(tgt_sentences)} строк)")
        
        # Проверка соответствия количества строк
        if len(src_sentences) != len(tgt_sentences):
            raise ValueError("Количество строк в параллельных файлах не совпадает!")
        
        # Создание директорий если нужно
        os.makedirs(os.path.dirname(src_output), exist_ok=True)
        os.makedirs(os.path.dirname(tgt_output), exist_ok=True)
        
        # Сохранение файлов
        with open(src_output, 'w', encoding=encoding) as f:
            for sentence in src_sentences:
                f.write(sentence + '\n')
        
        with open(tgt_output, 'w', encoding=encoding) as f:
            for sentence in tgt_sentences:
                f.write(sentence + '\n')
        
        print("Корпус успешно сохранен")
