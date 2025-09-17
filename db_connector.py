import pandas as pd
import numpy as np
import logging
import psycopg2
import io
import time
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, ProgrammingError
import pyodbc

class BaseDBConnector:
    def __init__(self, params, engine=None):
        self.params = params
        self.connection = None
        self.engine = engine # Добавляем engine как атрибут класса
        self._connect()

    def _connect(self):
        """Устанавливает соединение с GreenPlum."""
        try:
            self.connection = psycopg2.connect(**self.params)  # Присваиваем соединение
            self.connection.autocommit = True
            logging.info("Успешно подключились к GreenPlum.")
        except Exception as e:
            logging.error(f"Ошибка подключения к GreenPlum: {e}")
            self.connection = None  # Явно указываем, что соединение отсутствует
            raise

    def execute_query(self, query, max_retries=50, retry_delay=30):
        for attempt in range(1, max_retries + 1):
            try:
                # Проверяем состояние соединения
                if self.connection is None or self.connection.closed:
                    logging.warning("Соединение с базой потеряно. Переподключение...")
                    self._connect()

                with self.connection.cursor() as cursor:
                    cursor.execute(query)
                    logging.debug("DDL/DML запрос выполнен успешно.")
                    return

            except psycopg2.ProgrammingError as pe:
                # Синтаксические и прочие ошибки, связанные с запросом, повторять нет смысла
                logging.error(f"Синтаксическая ошибка или ошибка в запросе (попытка {attempt}): {pe}")
                raise

            except psycopg2.OperationalError as oe:
                # Ошибки уровня соединения/операционные ошибки — пробуем переподключиться и повторить
                logging.error(f"Операционная ошибка (попытка {attempt}): {oe}")
                if attempt < max_retries:
                    backoff = retry_delay * attempt
                    logging.info(f"Повторная попытка через {backoff} секунд...")
                    time.sleep(backoff)
                    self._connect()
                else:
                    # Последняя попытка — падаем с ошибкой
                    raise

            except Exception as e:
                # Другие неожиданные ошибки — повторять нет смысла,
                # так как неизвестно, исправятся ли они при повторной попытке
                logging.error(f"Неизвестная ошибка на попытке {attempt}: {e}")
                raise

    def insert_data(self, df, tablename='', sep='\t', max_retries=50, retry_delay=30):
        if df.empty:
            logging.warning("DataFrame пуст. Нечего вставлять")
            return 0

        csv_io = io.StringIO()
        df.to_csv(csv_io, sep=sep, header=False, index=False)
        csv_io.seek(0)

        for attempt in range(1, max_retries + 1):
            start_time = time.time()
            try:
                if self.connection is None or self.connection.closed:
                    logging.warning("Соединение с базой потеряно. Переподключение...")
                    self._connect()

                with self.connection.cursor() as cursor: 
                    cursor.copy_expert(f"COPY {tablename} FROM STDIN", csv_io)

                execution_time = round(time.time() - start_time, 3)
                logging.info(f"Успешно вставлено {len(df)} строк в таблицу {tablename} за {execution_time} сек.")
                return execution_time

            except psycopg2.ProgrammingError as pe:
                # Синтаксические ошибки, ошибки в запросе COPY и пр.
                logging.error(f"Синтаксическая/логическая ошибка вставки (попытка {attempt}): {pe}")
                # Не имеет смысла повторять, т.к. ошибка не устранится сама
                raise

            except psycopg2.OperationalError as oe:
                # Ошибки соединения - попробуем повторить попытку
                logging.error(f"Операционная ошибка вставки данных (попытка {attempt}): {oe}")
                if attempt < max_retries:
                    # Переподключаемся и пробуем заново
                    self._connect()
                    try:
                        with self.connection.cursor() as cleanup_cursor:
                            cleanup_cursor.execute(f"TRUNCATE TABLE {tablename}")
                            logging.info(f"Таблица {tablename} очищена после сбоя.")
                    except Exception as cleanup_error:
                        logging.error(f"Ошибка очистки таблицы {tablename}: {cleanup_error}")
                    backoff = retry_delay * attempt
                    logging.info(f"Повторная попытка вставки через {backoff} секунд...")
                    time.sleep(backoff)
                    csv_io.seek(0)
                else:
                    raise

            except Exception as e:
                # Любые другие неожиданные ошибки
                logging.error(f"Неизвестная ошибка при вставке данных (попытка {attempt}): {e}")
                # В таких случаях повторять обычно нет смысла, т.к. ошибка неизвестна
                raise

    def insert_data_executemany(self, df, tablename='', chunk_size=50000):
        """
        Вставляет данные из DataFrame в таблицу, используя executemany (обычные INSERT).
        """
        if df.empty:
            logging.warning("DataFrame пуст. Нечего вставлять.")
            return 0

        start_time = time.time()

        # Преобразуем DataFrame в список кортежей с преобразованием типов
        all_records = [tuple(map(lambda x: x.item() if isinstance(x, (np.integer, np.floating)) else x, row))
                    for row in df.to_records(index=False)]

        inserted_rows = 0
        with self.connection.cursor() as cursor:
            for i in range(0, len(all_records), chunk_size):
                chunk = all_records[i:i+chunk_size]

                # Формируем плейсхолдеры (%s, %s, ...) по количеству колонок df
                placeholders = ", ".join(["%s"] * df.shape[1])
                # Формируем INSERT-запрос
                insert_sql = f"INSERT INTO {tablename} VALUES ({placeholders})"
                cursor.executemany(insert_sql, chunk)
                inserted_rows += len(chunk)

        elapsed = round(time.time() - start_time, 3)
        logging.info(f"Вставлено {inserted_rows} строк в таблицу {tablename} за {elapsed} секунд.")
                
    def to_sql_list(self, series, quotes=False):
        """
        Преобразует список в SQL-список для использования в запросах.
        """
        if quotes:
            return ", ".join(f"'{str(item)}'" for item in series)
        return ", ".join(str(i) for i in list(series))
                
    def gp(self, query, max_retries=50, retry_delay=30, display_result=False): 
        for attempt in range(1, max_retries + 1):
            try:
                result = pd.read_sql_query(query, self.engine)
                if display_result:
                    print(result.to_string(index=False))
                return result

            except ProgrammingError as pe:
                # Синтаксическая ошибка в запросе
                logging.error(f"Синтаксическая/логическая ошибка SELECT (попытка {attempt}): {pe}")
                # Повторять бессмысленно
                raise

            except OperationalError as oe:
                # Проблема соединения, попробуем ещё раз
                logging.error(f"Операционная ошибка SELECT (попытка {attempt}): {oe}")
                if attempt < max_retries:
                    backoff = retry_delay * attempt
                    logging.info(f"Повторная попытка через {backoff} секунд...")
                    time.sleep(backoff)
                else:
                    # Кончились попытки
                    raise

            except Exception as e:
                # Любая иная ошибка, повторять обычно нет смысла
                logging.error(f"Неизвестная ошибка SELECT (попытка {attempt}): {e}")
                raise

    def close(self):
        """Закрывает соединение с базой данных."""
        if self.connection:
            self.connection.close()
            logging.info("Соединение с базой данных закрыто.")

class GreenPlumConnector(BaseDBConnector):
    def _connect(self):
        """Устанавливает соединение с GreenPlum."""
        try:
            self.connection = psycopg2.connect(**self.params)
            self.connection.autocommit = True
            logging.info("Успешно подключились к GreenPlum.")
        except Exception as e:
            logging.error(f"Ошибка подключения к GreenPlum: {e}")
            self.connection = None
            raise
    
    @staticmethod
    def reduce_mem_usage(df, verbose=True):
        import numpy as np
        """
        Уменьшает объем памяти, занимаемой DataFrame,
        автоматически понижая разряды числовых типов.
        """
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type not in [object, 'category']:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                else:
                    df[col] = df[col].astype(np.float32)
            # Если в данных есть категориальные строки
            # elif col_type == object:
            #     df[col] = df[col].astype('category')
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose:
            print(f'Память до уменьшения: {start_mem:.2f} MB')
            print(f'Память после уменьшения: {end_mem:.2f} MB')
            print(f'Уменьшили на {100 * (start_mem - end_mem) / start_mem:.1f}%')
        return df

# teradata connector

def teradata(script, odbc_connection, max_retries=50, retry_delay=30):
    attempt = 0
    while attempt < max_retries:
        try:
            with pyodbc.connect(odbc_connection, autocommit=True) as conn:
                df = pd.read_sql_query(script, conn)
            return df
        
        except pyodbc.OperationalError as oe:
            attempt += 1
            logging.error(f"Операционная ошибка при SELECT (попытка {attempt}): {oe}")
            if attempt < max_retries:
                backoff = retry_delay * attempt
                logging.info(f"Переподключение через {backoff} секунд...")
                time.sleep(backoff)
            else:
                raise
        except Exception as e:
            logging.error(f"Неизвестная ошибка при SELECT: {e}")
            raise
        
def execute_ddl_teradata(sql_script, odbc_connection, max_retries=50, retry_delay=30):
    attempt = 0
    while attempt < max_retries:
        try:
            with pyodbc.connect(odbc_connection, autocommit=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql_script)
            return True
    
        except pyodbc.OperationalError as oe:
            attempt += 1
            logging.error(f"Операционная ошибка при выполнении DDL (попытка {attempt}): {oe}")
            if attempt < max_retries:
                backoff = retry_delay * attempt
                logging.info(f"Переподключение через {backoff} секунд...")
                time.sleep(backoff)
            else:
                raise
        except Exception as e:
            logging.error(f"Неизвестная ошибка при DDL: {e}")
            raise
        
def execute_many(sql_script, data, odbc_connection, max_retries=50, retry_delay=30):
    attempt = 0
    while attempt < max_retries:
        try:
            with pyodbc.connect(odbc_connection, autocommit=True) as conn:
                with conn.cursor() as cursor:
                    cursor.executemany(sql_script, data)
            return True

        except pyodbc.OperationalError as oe:
            attempt += 1
            logging.error(f"Ошибка при executemany (попытка {attempt}): {oe}")
            if attempt < max_retries:
                backoff = retry_delay * attempt
                logging.info(f"Переподключение через {backoff} секунд...")
                time.sleep(backoff)
            else:
                raise
        except Exception as e:
            logging.error(f"Неизвестная ошибка при executemany: {e}")
            raise