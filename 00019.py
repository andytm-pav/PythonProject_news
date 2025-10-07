import os
import sys
import time
import requests
import feedparser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import re
import threading
from flask import Flask, render_template_string, request, jsonify
import json
import traceback
from urllib.request import urlretrieve
import zipfile
import tempfile
from scipy.optimize import minimize
import logging
from logging.handlers import RotatingFileHandler
import hashlib
from functools import wraps
import asyncio
#import aiohttp
import concurrent.futures

from news_analyzer import NewsAnalyzer
from news_collector import NewsCollector
from stock_data_collector import StockDataCollector
from virtual_portfolio import VirtualPortfolio
from web_interface import WebInterface

# ==================== КОНФИГУРАЦИЯ И НАСТРОЙКА ЛОГГИРОВАНИЯ ====================
warnings.filterwarnings('ignore')


# Настройка логирования
def setup_logging():
    logger = logging.getLogger('StockAnalysis')
    logger.setLevel(logging.INFO)

    # Форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    # File handler
    file_handler = RotatingFileHandler(
        'stock_analysis.log', maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


# ==================== КОНФИГУРАЦИЯ БЕЗОПАСНОСТИ ====================
class SecurityConfig:
    """Конфигурация безопасности и ограничений"""

    # Ограничения портфеля
    MAX_DRAWDOWN = 0.15  # Максимальная просадка 15%
    SECTOR_LIMITS = 0.25  # Не более 25% в один сектор
    POSITION_SIZING = 0.1  # Не более 10% в одну позицию
    STOP_LOSS = 0.08  # Стоп-лосс 8%

    # Ограничения API
    MAX_REQUESTS_PER_MINUTE = 30
    REQUEST_TIMEOUT = 30
    RETRY_ATTEMPTS = 3

    # Настройки модели
    MIN_TRAINING_SAMPLES = 100
    MIN_HISTORY_DAYS = 252  # 1 год данных
    WALK_FORWARD_WINDOW = 63  # 3 месяца для валидации


# ==================== КЛАСС ДЛЯ ОБРАБОТКИ ОШИБОК ====================
class ErrorHandler:
    """Универсальный обработчик ошибок с детализированным логированием"""

    @staticmethod
    def handle_data_error(func):
        """Декоратор для обработки ошибок работы с данными"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except pd.errors.EmptyDataError as e:
                logger.error(f"Пустые данные в {func.__name__}: {e}")
                return None
            except pd.errors.ParserError as e:
                logger.error(f"Ошибка парсинга в {func.__name__}: {e}")
                return None
            except ValueError as e:
                logger.error(f"Ошибка значений в {func.__name__}: {e}")
                return None
            except Exception as e:
                logger.error(f"Неожиданная ошибка в {func.__name__}: {e}")
                return None

        return wrapper

    @staticmethod
    def handle_network_error(func):
        """Декоратор для обработки сетевых ошибок"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(SecurityConfig.RETRY_ATTEMPTS):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.Timeout as e:
                    logger.warning(f"Таймаут попытка {attempt + 1} в {func.__name__}: {e}")
                    if attempt == SecurityConfig.RETRY_ATTEMPTS - 1:
                        logger.error(f"Все попытки таймаута исчерпаны в {func.__name__}")
                        return None
                    time.sleep(2 ** attempt)  # Exponential backoff
                except requests.exceptions.ConnectionError as e:
                    logger.warning(f"Ошибка соединения попытка {attempt + 1} в {func.__name__}: {e}")
                    if attempt == SecurityConfig.RETRY_ATTEMPTS - 1:
                        logger.error(f"Все попытки соединения исчерпаны в {func.__name__}")
                        return None
                    time.sleep(2 ** attempt)
                except requests.exceptions.RequestException as e:
                    logger.error(f"Ошибка запроса в {func.__name__}: {e}")
                    return None
                except Exception as e:
                    logger.error(f"Неожиданная сетевая ошибка в {func.__name__}: {e}")
                    return None

        return wrapper

    @staticmethod
    def handle_model_error(func):
        """Декоратор для обработки ошибок ML моделей"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValueError as e:
                logger.error(f"Ошибка данных модели в {func.__name__}: {e}")
                return None
            except AttributeError as e:
                logger.error(f"Ошибка атрибута модели в {func.__name__}: {e}")
                return None
            except Exception as e:
                logger.error(f"Неожиданная ошибка модели в {func.__name__}: {e}")
                return None

        return wrapper


# ==================== КОНФИГУРАЦИЯ ДАННЫХ ====================
RUSSIAN_SECTORS = {
    'Энергетика': ['GAZP', 'LKOH', 'ROSN', 'SNGS', 'TATN'],
    'Индустриальный': ['ALRS', 'CHMK', 'MTLR', 'TRNFP'],
    'Базовые ресурсы и материалы': ['GMKN', 'NLMK', 'MAGN', 'PLZL', 'RUAL'],
    'Розничная и оптовая торговля': ['FIVE', 'DSKY', 'LNTA', 'OZON'],
    'Медицина, фармацевтика, охрана здоровья': ['POLY', 'RGSS', 'YNDX'],
    'Потребительский': ['GCHE', 'UPRO', 'WUSH'],
    'Телекоммуникации': ['MTSS', 'RTKM', 'MGNT'],
    'Химия и нефтехимия': ['PHOR', 'AKRN', 'ODVA'],
    'Электроэнергетика': ['FEES', 'HYDR', 'IRAO'],
    'Строительство': ['PIKK', 'LSRG', 'UNAC'],
    'Транспорт': ['AFLT', 'NMTP', 'TRMK']
}

RUSSIAN_FEEDS = {
    "Ведомости": "https://www.vedomosti.ru/rss/news",
    "Коммерсант": "https://www.kommersant.ru/RSS/news.xml",
    "Интерфакс": "https://www.interfax.ru/rss.asp",
    "ТАСС": "https://tass.ru/rss/v2.xml",
    "Лента.ру": "https://lenta.ru/rss/news"
}


# ==================== УЛУЧШЕННЫЙ КЛАСС ИИ МОДЕЛИ С ОБРАБОТКОЙ ОШИБОК ====================
class EnhancedStockAIModel:
    def __init__(self, model_path='enhanced_stock_ai_model_v2.joblib'):
        self.model_path = model_path
        self.model = None
        self.probabilistic_model = None
        self.advanced_model = None
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_selector = None
        self.is_trained = False
        self.training_history = []
        self.model_metadata = {}
        self.logger = logger
        self.benchmark_returns = {}

        self._initialize_model_with_validation()

    def _initialize_model_with_validation(self):
        """Инициализация модели с валидацией данных"""
        self.logger.info("🔄 Инициализация ИИ моделей с валидацией...")

        if not self._try_load_existing_model():
            self.logger.info("📝 Создание новой модели...")
            self._create_new_models_with_validation()

    def _try_load_existing_model(self):
        """Попытка загрузки существующей модели с валидацией"""
        try:
            if not os.path.exists(self.model_path):
                self.logger.warning("Файл модели не существует")
                return False

            model_data = joblib.load(self.model_path)
            return self._validate_and_load_model(model_data)

        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки модели: {e}")
            return False

    def _validate_and_load_model(self, model_data):
        """Валидация и загрузка модели"""
        try:
            required_keys = ['main_model', 'probabilistic_model', 'advanced_model',
                             'scaler', 'imputer', 'model_metadata']

            missing_keys = [key for key in required_keys if key not in model_data]
            if missing_keys:
                self.logger.error(f"Отсутствующие ключи в модели: {missing_keys}")
                return False

            # Проверка версии модели
            metadata = model_data.get('model_metadata', {})
            if metadata.get('model_type') != 'EnhancedStockAIModel_v2':
                self.logger.warning("Устаревшая версия модели, создаем новую")
                return False

            self.model = model_data['main_model']
            self.probabilistic_model = model_data['probabilistic_model']
            self.advanced_model = model_data['advanced_model']
            self.scaler = model_data['scaler']
            self.imputer = model_data['imputer']
            self.is_trained = model_data.get('is_trained', False)
            self.training_history = model_data.get('training_history', [])
            self.model_metadata = metadata

            # Валидация обученности модели
            if self.is_trained and self.model_metadata.get('total_training_samples',
                                                           0) < SecurityConfig.MIN_TRAINING_SAMPLES:
                self.logger.warning("Модель обучена на недостаточном количестве данных")
                self.is_trained = False

            self.logger.info("✅ Модель успешно загружена и валидирована")
            return True

        except Exception as e:
            self.logger.error(f"❌ Ошибка валидации модели: {e}")
            return False

    def _create_new_models_with_validation(self):
        """Создание новых моделей с валидацией параметров"""
        try:
            # Основная модель с оптимизированными параметрами
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )

            # Вероятностная модель
            self.probabilistic_model = BayesianRidge(
                n_iter=300,
                tol=1e-4,
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6
            )

            # Улучшенная модель
            self.advanced_model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                verbose=0
            )

            self.imputer = SimpleImputer(strategy='median')
            self.scaler = RobustScaler()

            self.is_trained = False
            self.model_metadata = {
                'creation_date': datetime.now().isoformat(),
                'total_training_samples': 0,
                'training_sessions': 0,
                'last_training': None,
                'model_type': 'EnhancedStockAIModel_v2',
                'feature_count': 0,
                'validation_score': 0.0
            }

            self.logger.info("✅ Новые модели созданы с валидацией параметров")

        except Exception as e:
            self.logger.error(f"❌ Ошибка создания моделей: {e}")
            # Резервные простые модели
            self.model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.probabilistic_model = BayesianRidge()
            self.advanced_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
            self.is_trained = False

    @ErrorHandler.handle_data_error
    def _validate_features(self, features_dict):
        """Расширенная валидация признаков"""
        if not features_dict:
            self.logger.warning("Пустой словарь признаков")
            return {}

        validated_features = {}
        validation_errors = []

        for ticker, features in features_dict.items():
            try:
                if not features:
                    validation_errors.append(f"Пустые признаки для {ticker}")
                    continue

                # Преобразование в numpy array
                feature_array = np.array(list(features.values()), dtype=np.float64)

                # Проверка на NaN
                if np.any(np.isnan(feature_array)):
                    validation_errors.append(f"NaN значения в {ticker}")
                    continue

                # Проверка на бесконечности
                if np.any(np.isinf(feature_array)):
                    validation_errors.append(f"Бесконечные значения в {ticker}")
                    continue

                # Проверка на нулевую дисперсию
                if np.std(feature_array) < 1e-8:
                    validation_errors.append(f"Нулевая дисперсия в {ticker}")
                    continue

                # Проверка на выбросы (3 sigma rule)
                z_scores = np.abs((feature_array - np.mean(feature_array)) / np.std(feature_array))
                if np.any(z_scores > 5):
                    self.logger.warning(f"Обнаружены выбросы в {ticker}, но признаки сохранены")

                validated_features[ticker] = features

            except (ValueError, TypeError) as e:
                validation_errors.append(f"Ошибка преобразования {ticker}: {e}")
                continue

        if validation_errors:
            self.logger.warning(f"Проблемы валидации: {validation_errors[:5]}")  # Логируем первые 5 ошибок

        self.logger.info(f"✅ Валидировано {len(validated_features)} из {len(features_dict)} наборов признаков")
        return validated_features

    @ErrorHandler.handle_data_error
    def calculate_technical_indicators(self, price_series):
        """Расчет технических индикаторов с расширенной валидацией"""
        if price_series is None or len(price_series) < 50:  # Увеличили минимальную длину
            self.logger.warning(
                f"Недостаточно данных для расчета индикаторов: {len(price_series) if price_series else 0} точек")
            return {}

        try:
            # Валидация входных данных
            if price_series.isna().any():
                self.logger.warning("Обнаружены NaN в ценовой серии, заполняем...")
                price_series = price_series.ffill().bfill()

            if (price_series <= 0).any():
                self.logger.error("Обнаружены неположительные цены")
                return {}

            returns = price_series.pct_change().dropna()
            if len(returns) < 20:
                return {}

            # Скользящие средные с проверкой достаточности данных
            windows = [5, 20, 50, 100]
            sma_values = {}
            for window in windows:
                if len(price_series) >= window:
                    sma = price_series.rolling(window).mean()
                    sma_values[f'sma_{window}'] = sma.iloc[-1] if not sma.isna().iloc[-1] else price_series.iloc[-1]
                else:
                    sma_values[f'sma_{window}'] = price_series.iloc[-1]

            # RSI с обработкой edge cases
            delta = price_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

            # Избегаем деления на ноль
            rs = np.where(loss != 0, gain / loss, 1.0)
            rsi = 100 - (100 / (1 + rs))
            rsi_value = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

            # Волатильность с разными периодами
            volatility_short = returns.tail(20).std() if len(returns) >= 20 else 0.02
            volatility_medium = returns.tail(50).std() if len(returns) >= 50 else volatility_short
            volatility_long = returns.std() if len(returns) >= 100 else volatility_medium

            # Моментум с защитой от деления на ноль
            periods = [22, 66, 132]  # 1, 3, 6 месяцев
            momentum_values = {}
            for period in periods:
                if len(price_series) > period:
                    prev_price = price_series.iloc[-period - 1]
                    if prev_price > 0:
                        momentum_values[f'momentum_{period}'] = (price_series.iloc[-1] / prev_price - 1)
                    else:
                        momentum_values[f'momentum_{period}'] = 0.0
                else:
                    momentum_values[f'momentum_{period}'] = 0.0

            # MACD
            if len(price_series) >= 26:
                ema_12 = price_series.ewm(span=12).mean()
                ema_26 = price_series.ewm(span=26).mean()
                macd = ema_12 - ema_26
                macd_signal = macd.ewm(span=9).mean()
                macd_histogram = macd - macd_signal
                macd_value = macd_histogram.iloc[-1] if not pd.isna(macd_histogram.iloc[-1]) else 0.0
            else:
                macd_value = 0.0

            # Сбор всех индикаторов
            indicators = {
                'sma_ratio_5_20': sma_values['sma_5'] / sma_values['sma_20'] if sma_values['sma_20'] != 0 else 1.0,
                'sma_ratio_20_50': sma_values['sma_20'] / sma_values['sma_50'] if sma_values['sma_50'] != 0 else 1.0,
                'sma_ratio_50_100': sma_values['sma_50'] / sma_values['sma_100'] if sma_values['sma_100'] != 0 else 1.0,
                'price_vs_sma20': price_series.iloc[-1] / sma_values['sma_20'] if sma_values['sma_20'] != 0 else 1.0,
                'rsi': float(rsi_value),
                'volatility_ratio_short': volatility_short / volatility_long if volatility_long != 0 else 1.0,
                'volatility_ratio_medium': volatility_medium / volatility_long if volatility_long != 0 else 1.0,
                'macd_histogram': float(macd_value),
                'volatility': float(volatility_long),
                'current_price': float(price_series.iloc[-1])
            }

            # Добавляем моментум
            indicators.update({k: float(v) for k, v in momentum_values.items()})

            # Финальная проверка на валидность
            for key, value in indicators.items():
                if np.isnan(value) or np.isinf(value):
                    self.logger.warning(f"Некорректное значение индикатора {key}: {value}")
                    return {}

            return indicators

        except Exception as e:
            self.logger.error(f"❌ Ошибка расчета индикаторов: {e}")
            return {}

    @ErrorHandler.handle_data_error
    def create_advanced_features(self, prices_df, news_sentiment):
        """Создание расширенных признаков с комплексной валидацией"""
        if prices_df is None or prices_df.empty:
            self.logger.error("Пустой DataFrame цен")
            return {}

        features = {}
        successful_features = 0
        failed_features = 0

        for ticker in prices_df.columns:
            try:
                if ticker not in prices_df:
                    failed_features += 1
                    continue

                price_series = prices_df[ticker].dropna()
                if len(price_series) < 50:  # Увеличили минимальное требование
                    failed_features += 1
                    continue

                # Технические индикаторы
                technical_features = self.calculate_technical_indicators(price_series)
                if not technical_features:
                    failed_features += 1
                    continue

                # Базовые признаки
                sector = self.get_sector_for_ticker(ticker)
                sector_sentiment = news_sentiment.get(sector, 0)

                # Дополнительные ценовые метрики
                price_stats = {
                    'price_range_20': (price_series.tail(20).max() - price_series.tail(20).min()) / price_series.tail(
                        20).mean(),
                    'volume_profile': price_series.rolling(20).std().iloc[
                                          -1] / price_series.mean() if price_series.mean() != 0 else 0,
                    'trend_strength': self._calculate_trend_strength(price_series)
                }

                feature_set = {
                    **technical_features,
                    **price_stats,
                    'sector_sentiment': float(sector_sentiment),
                    'days_of_data': len(price_series)
                }

                features[ticker] = feature_set
                successful_features += 1

            except Exception as e:
                self.logger.error(f"❌ Ошибка создания признаков для {ticker}: {e}")
                failed_features += 1
                continue

        self.logger.info(f"📊 Создано признаков: {successful_features} успешно, {failed_features} неудачно")

        # Валидация всех признаков
        validated_features = self._validate_features(features)
        return validated_features

    def _calculate_trend_strength(self, price_series):
        """Расчет силы тренда"""
        if len(price_series) < 20:
            return 0.0

        try:
            # Linear regression для определения тренда
            x = np.arange(len(price_series)).reshape(-1, 1)
            y = price_series.values.reshape(-1, 1)

            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(x, y)
            trend_slope = reg.coef_[0][0]
            r_squared = reg.score(x, y)

            return float(trend_slope * r_squared)
        except:
            return 0.0

    @ErrorHandler.handle_data_error
    def prepare_training_data(self, prices_df, news_sentiment_by_date, forecast_days=10):
        """Подготовка данных для обучения с расширенной валидацией"""
        self.logger.info("📋 Подготовка данных для обучения...")

        if prices_df is None or prices_df.empty:
            self.logger.error("❌ Пустой DataFrame цен для обучения")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Валидация входных данных
        if prices_df.isna().any().any():
            self.logger.warning("Обнаружены NaN в ценах, заполняем...")
            prices_df = prices_df.ffill().bfill()
            # Удаляем столбцы с оставшимися NaN
            prices_df = prices_df.dropna(axis=1, how='any')

        if prices_df.empty:
            self.logger.error("❌ DataFrame стал пустым после обработки NaN")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Получаем текущий сентимент
        current_sentiment = {}
        if news_sentiment_by_date:
            try:
                latest_date = max(news_sentiment_by_date.keys())
                current_sentiment = news_sentiment_by_date[latest_date]
            except Exception as e:
                self.logger.warning(f"Ошибка получения сентимента: {e}")

        features_dict = self.create_advanced_features(prices_df, current_sentiment)

        if not features_dict:
            self.logger.error("❌ Нет валидных признаков для обучения")
            return np.array([]), np.array([]), np.array([]), np.array([])

        X_basic, X_advanced, y_basic, y_advanced = [], [], [], []
        successful_targets = 0
        failed_targets = 0

        for ticker, features in features_dict.items():
            try:
                if not features or len(features) == 0:
                    failed_targets += 1
                    continue

                if ticker not in prices_df.columns:
                    failed_targets += 1
                    continue

                price_series = prices_df[ticker].dropna()
                if len(price_series) <= forecast_days:
                    failed_targets += 1
                    continue

                # РЕАЛЬНАЯ целевая переменная - будущая доходность
                current_price = price_series.iloc[-1]
                future_idx = -forecast_days - 1

                if len(price_series) > abs(future_idx):
                    future_price = price_series.iloc[future_idx]

                    # Проверка валидности цен
                    if current_price <= 0 or future_price <= 0:
                        failed_targets += 1
                        continue

                    actual_return = (current_price / future_price - 1)

                    # Проверка на разумность доходности
                    if abs(actual_return) > 2.0:  # Фильтр экстремальных значений
                        self.logger.warning(f"Экстремальная доходность для {ticker}: {actual_return:.3f}")
                        failed_targets += 1
                        continue

                    feature_vector = list(features.values())

                    # Финальная проверка feature vector
                    if (len(feature_vector) == len(features) and
                            not any(np.isnan(feature_vector)) and
                            not any(np.isinf(feature_vector)) and
                            not np.isnan(actual_return)):

                        X_basic.append(feature_vector)
                        X_advanced.append(feature_vector)
                        y_basic.append(actual_return)
                        y_advanced.append(actual_return)
                        successful_targets += 1
                    else:
                        failed_targets += 1
                else:
                    failed_targets += 1

            except Exception as e:
                self.logger.error(f"❌ Ошибка подготовки данных для {ticker}: {e}")
                failed_targets += 1
                continue

        self.logger.info(f"🎯 Подготовлено целей: {successful_targets} успешно, {failed_targets} неудачно")

        if successful_targets < SecurityConfig.MIN_TRAINING_SAMPLES:
            self.logger.warning(
                f"⚠️ Мало примеров для обучения: {successful_targets} < {SecurityConfig.MIN_TRAINING_SAMPLES}")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Конвертация в numpy arrays с проверкой
        try:
            X_basic_array = np.array(X_basic, dtype=np.float64)
            X_advanced_array = np.array(X_advanced, dtype=np.float64)
            y_basic_array = np.array(y_basic, dtype=np.float64)
            y_advanced_array = np.array(y_advanced, dtype=np.float64)

            # Финальная проверка размерностей
            if (X_basic_array.shape[0] == y_basic_array.shape[0] == successful_targets and
                    X_advanced_array.shape[0] == y_advanced_array.shape[0] == successful_targets):

                self.logger.info(f"✅ Данные подготовлены: {X_basic_array.shape} features, {successful_targets} samples")
                return X_basic_array, X_advanced_array, y_basic_array, y_advanced_array
            else:
                self.logger.error("❌ Несоответствие размерностей данных")
                return np.array([]), np.array([]), np.array([]), np.array([])

        except Exception as e:
            self.logger.error(f"❌ Ошибка конвертации в numpy: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])

    # ... остальные методы класса с аналогичной обработкой ошибок ...

    @ErrorHandler.handle_model_error
    def train_models(self, prices_df, news_sentiment_by_date, incremental=True):
        """Обучение моделей с расширенной валидацией и обработкой ошибок"""
        self.logger.info("🔄 Запуск улучшенного обучения моделей...")

        try:
            if self.model is None:
                self._create_new_models_with_validation()

            # Подготовка данных с валидацией
            X_basic, X_advanced, y_basic, y_advanced = self.prepare_training_data(
                prices_df, news_sentiment_by_date
            )

            # Проверка достаточности данных
            if len(X_basic) < SecurityConfig.MIN_TRAINING_SAMPLES:
                self.logger.error(
                    f"❌ Недостаточно данных для обучения: {len(X_basic)} < {SecurityConfig.MIN_TRAINING_SAMPLES}")
                return False

            self.logger.info(f"🎯 Обучение на {len(X_basic)} валидных примерах...")

            # Walk-forward validation
            tscv = TimeSeriesSplit(n_splits=5)
            validation_scores = []

            for train_idx, test_idx in tscv.split(X_basic):
                try:
                    X_train, X_test = X_basic[train_idx], X_basic[test_idx]
                    y_train, y_test = y_basic[train_idx], y_basic[test_idx]

                    # Обучение и оценка
                    self.model.fit(X_train, y_train)
                    y_pred = self.model.predict(X_test)
                    score = mean_squared_error(y_test, y_pred)
                    validation_scores.append(score)

                except Exception as e:
                    self.logger.warning(f"Ошибка в fold валидации: {e}")
                    continue

            if validation_scores:
                avg_score = np.mean(validation_scores)
                self.model_metadata['validation_score'] = avg_score
                self.logger.info(f"📊 Средняя MSE валидации: {avg_score:.6f}")

            # Финальное обучение на всех данных
            try:
                X_processed = self._safe_fit_transform(X_basic, incremental)
                X_advanced_processed = self._safe_fit_transform(X_advanced, incremental)

                if self.is_trained and incremental:
                    self.logger.info("📚 Дообучение моделей...")
                    self.model.fit(X_processed, y_basic)
                    self.probabilistic_model.fit(X_processed, y_basic)
                    self.advanced_model.fit(X_advanced_processed, y_advanced)
                else:
                    self.logger.info("🎓 Первоначальное обучение моделей...")
                    self.model.fit(X_processed, y_basic)
                    self.probabilistic_model.fit(X_processed, y_basic)
                    self.advanced_model.fit(X_advanced_processed, y_advanced)

            except Exception as e:
                self.logger.error(f"❌ Ошибка обучения моделей: {e}")
                return False

            # Обновление метаданных
            self.is_trained = True
            self._update_training_metadata(len(X_basic))
            self.model_metadata['feature_count'] = X_basic.shape[1] if len(X_basic.shape) > 1 else 0

            # Сохранение моделей
            if not self._save_models():
                self.logger.error("❌ Не удалось сохранить модели")
                return False

            avg_return = np.mean(y_advanced) if len(y_advanced) > 0 else 0
            self.logger.info(f"✅ Обучение завершено. Средняя доходность: {avg_return:.4f}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Критическая ошибка обучения: {e}")
            traceback.print_exc()
            return False

    def _safe_fit_transform(self, X, incremental=False):
        """Безопасное преобразование данных с расширенной обработкой ошибок"""
        if len(X) == 0:
            return X

        try:
            # Заменяем оставшиеся NaN через импьютер
            if not self.is_trained or not incremental:
                X_imputed = self.imputer.fit_transform(X)
            else:
                X_imputed = self.imputer.transform(X)

            # Проверка результата импьютации
            if np.any(np.isnan(X_imputed)):
                self.logger.warning("NaN остались после импьютации, применяем дополнительную обработку")
                X_imputed = np.nan_to_num(X_imputed, nan=0.0)

            # Масштабирование
            if not self.is_trained or not incremental:
                X_scaled = self.scaler.fit_transform(X_imputed)
            else:
                X_scaled = self.scaler.transform(X_imputed)

            # Финальная проверка
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                self.logger.error("Обнаружены NaN/Inf после масштабирования")
                return X_imputed  # Возвращаем немасштабированные, но очищенные данные

            return X_scaled

        except Exception as e:
            self.logger.error(f"❌ Ошибка преобразования данных: {e}")
            return X  # Возвращаем исходные данные в случае ошибки


# ==================== КЛАСС РИСК-МЕНЕДЖМЕНТА ====================
class RiskManager:
    """Расширенный менеджер рисков с мониторингом в реальном времени"""

    def __init__(self):
        self.max_drawdown = SecurityConfig.MAX_DRAWDOWN
        self.sector_limits = SecurityConfig.SECTOR_LIMITS
        self.position_sizing = SecurityConfig.POSITION_SIZING
        self.stop_loss = SecurityConfig.STOP_LOSS
        self.risk_free_rate = 0.05  # 5% безрисковая ставка
        self.logger = logger

    def validate_portfolio_weights(self, weights, prices_df, current_positions=None):
        """Валидация весов портфеля с учетом рисков"""
        if not weights:
            self.logger.warning("Пустые веса портфеля")
            return {}

        validated_weights = {}
        total_weight = 0
        sector_exposure = {}

        for ticker, weight in weights.items():
            try:
                # Проверка существования тикера
                if ticker not in prices_df.columns:
                    self.logger.warning(f"Тикер {ticker} отсутствует в данных, пропускаем")
                    continue

                # Проверка лимита позиции
                if weight > self.position_sizing:
                    self.logger.warning(f"Превышен лимит позиции для {ticker}: {weight:.3f} > {self.position_sizing}")
                    weight = self.position_sizing

                # Проверка на отрицательные веса
                if weight < 0:
                    self.logger.warning(f"Отрицательный вес для {ticker}: {weight:.3f}")
                    continue

                # Расчет экспозиции по секторам
                sector = self._get_sector_for_ticker(ticker)
                sector_exposure[sector] = sector_exposure.get(sector, 0) + weight

                # Проверка лимита сектора
                if sector_exposure[sector] > self.sector_limits:
                    self.logger.warning(
                        f"Превышен лимит сектора {sector}: {sector_exposure[sector]:.3f} > {self.sector_limits}")
                    available_weight = self.sector_limits - (sector_exposure[sector] - weight)
                    weight = min(weight, available_weight)

                if weight > 0:
                    validated_weights[ticker] = weight
                    total_weight += weight

            except Exception as e:
                self.logger.error(f"Ошибка валидации веса для {ticker}: {e}")
                continue

        # Нормализация весов
        if total_weight > 0:
            validated_weights = {k: v / total_weight for k, v in validated_weights.items()}
            self.logger.info(f"✅ Валидированы веса для {len(validated_weights)} активов")
        else:
            self.logger.warning("Нет валидных весов после проверки рисков")

        return validated_weights

    def _get_sector_for_ticker(self, ticker):
        """Получение сектора для тикера"""
        for sector, tickers in RUSSIAN_SECTORS.items():
            if ticker in tickers:
                return sector
        return 'Общие'


# ==================== ОСНОВНОЙ КЛАСС СИСТЕМЫ С ОБРАБОТКОЙ ОШИБОК ====================
class EnhancedStockAnalysisSystem:
    def __init__(self):
        self.logger = logger
        self.news_collector = NewsCollector()
        self.stock_collector = StockDataCollector()
        self.news_analyzer = NewsAnalyzer()
        self.enhanced_ai_model = EnhancedStockAIModel()
        self.portfolio = VirtualPortfolio()
        self.web_interface = WebInterface()
        self.risk_manager = RiskManager()

        self.news_backup_file = 'news_backup.csv'
        self.prices_backup_file = 'prices_backup.csv'
        self.recommendations_file = 'RECOM.csv'

        self.setup_system()

    def setup_system(self):
        """Настройка системы с обработкой ошибок инициализации"""
        try:
            # Проверка зависимостей
            self._check_dependencies()

            # Создание необходимых директорий
            self._create_directories()

            # Инициализация компонентов
            self._initialize_components()

            self.logger.info("✅ Система успешно инициализирована")

        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации системы: {e}")
            raise

    def _check_dependencies(self):
        """Проверка необходимых зависимостей"""
        required_libraries = [
            'pandas', 'numpy', 'sklearn', 'requests',
            'feedparser', 'flask', 'joblib', 'scipy'
        ]

        missing_libraries = []
        for lib in required_libraries:
            try:
                __import__(lib)
            except ImportError:
                missing_libraries.append(lib)

        if missing_libraries:
            error_msg = f"Отсутствуют библиотеки: {', '.join(missing_libraries)}"
            self.logger.error(error_msg)
            raise ImportError(error_msg)

    def _create_directories(self):
        """Создание необходимых директорий"""
        directories = ['data', 'models', 'logs', 'backups']
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                self.logger.warning(f"Не удалось создать директорию {directory}: {e}")

    def _initialize_components(self):
        """Инициализация компонентов системы"""
        # Проверка доступности моделей
        if not self.enhanced_ai_model.is_trained:
            self.logger.warning("Модель ИИ не обучена, требуется первоначальное обучение")

    @ErrorHandler.handle_network_error
    def run_enhanced_analysis_cycle(self):
        """Улучшенный цикл анализа с комплексной обработкой ошибок"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"🔍 Запуск улучшенного анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)

        try:
            # 1. Сбор данных с обработкой ошибок
            self.logger.info("📰 Шаг 1: Сбор новостей...")
            news_df = self.news_collector.fetch_all_news()
            if news_df is None or news_df.empty:
                self.logger.warning("Не удалось собрать новости, используем предыдущие данные")
                news_df = self._load_backup_news()

            # 2. Сбор данных об акциях
            self.logger.info("📊 Шаг 2: Сбор данных об акциях...")
            all_tickers = [ticker for sector_tickers in RUSSIAN_SECTORS.values() for ticker in sector_tickers]
            prices_df = self.stock_collector.get_stock_prices(all_tickers, days=252)  # 1 год данных

            if prices_df is None or prices_df.empty:
                self.logger.error("❌ Не удалось получить данные об акциях")
                return False

            # 3. Анализ новостей
            self.logger.info("🔍 Шаг 3: Анализ новостей...")
            news_recommendations = self.news_analyzer.predict_sector_sentiment(news_df)

            # 4. Обучение AI моделей
            self.logger.info("🎯 Шаг 4: Обучение AI моделей...")
            news_sentiment_by_date = {datetime.now().date(): news_recommendations}

            training_successful = self.enhanced_ai_model.train_models(
                prices_df, news_sentiment_by_date, incremental=True
            )

            if not training_successful:
                self.logger.warning("Обучение не удалось, используем базовые рекомендации")

            # 5. AI анализ и оптимизация портфеля
            self.logger.info("🧠 Шаг 5: AI анализ и оптимизация портфеля...")
            current_prices = prices_df.iloc[-1].to_dict()

            optimal_weights = {}
            if self.enhanced_ai_model.is_trained:
                ai_predictions = self.enhanced_ai_model.predict_returns(prices_df, news_recommendations)
                optimal_weights = self.enhanced_ai_model.optimize_portfolio(ai_predictions, prices_df)

                # Валидация весов через risk manager
                optimal_weights = self.risk_manager.validate_portfolio_weights(
                    optimal_weights, prices_df, self.portfolio.positions
                )

            # 6. Формирование рекомендаций
            enhanced_recommendations = self._enhance_recommendations_with_optimization(
                news_recommendations, optimal_weights
            )

            # 7. Сохранение и отображение результатов
            portfolio_value = self.portfolio.get_portfolio_value(current_prices)
            self.save_backup_data(news_df, prices_df, enhanced_recommendations)

            # Обновление веб-интерфейса
            model_info = self.enhanced_ai_model.get_model_status()
            self.web_interface.update_data(
                enhanced_recommendations, news_df, portfolio_value, optimal_weights, model_info
            )

            # 8. Вывод результатов
            self._print_enhanced_results(enhanced_recommendations, optimal_weights, portfolio_value)

            return True

        except Exception as e:
            self.logger.error(f"💥 Критическая ошибка в цикле анализа: {e}")
            traceback.print_exc()
            return False

    def _load_backup_news(self):
        """Загрузка резервных новостей"""
        try:
            if os.path.exists(self.news_backup_file):
                return pd.read_csv(self.news_backup_file)
        except Exception as e:
            self.logger.error(f"Ошибка загрузки резервных новостей: {e}")
        return pd.DataFrame()

    def run_continuous_analysis(self, interval_minutes=3):
        """Запуск непрерывного анализа с обработкой сбоев"""
        self.logger.info("🚀 Запуск улучшенной системы анализа российских акций с ИИ...")

        # Юридический дисклеймер
        self._print_legal_disclaimer()

        try:
            # Запуск веб-интерфейса в отдельном потоке
            web_thread = threading.Thread(target=self.web_interface.run_server)
            web_thread.daemon = True
            web_thread.start()

            cycle_count = 0
            consecutive_failures = 0
            max_consecutive_failures = 3

            while self.web_interface.is_running and consecutive_failures < max_consecutive_failures:
                try:
                    success = self.run_enhanced_analysis_cycle()
                    cycle_count += 1

                    if success:
                        consecutive_failures = 0
                        self.logger.info(f"♻️ Цикл {cycle_count} завершен. Ожидание {interval_minutes} минут...")

                        # Ожидание с проверкой флага остановки
                        wait_seconds = interval_minutes * 60
                        for i in range(wait_seconds):
                            if not self.web_interface.is_running:
                                break
                            time.sleep(1)
                    else:
                        consecutive_failures += 1
                        self.logger.error(
                            f"❌ Ошибка цикла {cycle_count}. Попытка {consecutive_failures}/{max_consecutive_failures}")
                        time.sleep(300)  # 5 минут при ошибке

                except KeyboardInterrupt:
                    self.logger.info("⚠️ Программа прервана пользователем")
                    break
                except Exception as e:
                    consecutive_failures += 1
                    self.logger.error(f"💥 Неожиданная ошибка: {e}")
                    time.sleep(300)

            if consecutive_failures >= max_consecutive_failures:
                self.logger.error("🛑 Достигнут лимит последовательных ошибок, остановка системы")

        except Exception as e:
            self.logger.error(f"💥 Критическая ошибка системы: {e}")
        finally:
            self.logger.info("🛑 Программа остановлена")

    def _print_legal_disclaimer(self):
        """Вывод юридического дисклеймера"""
        disclaimer = """
        ⚠️ ЮРИДИЧЕСКОЕ УВЕДОМЛЕНИЕ И ОТКАЗ ОТ ОТВЕТСТВЕННОСТИ

       
        """
        print(disclaimer)
        self.logger.info("Юридический дисклеймер отображен")


# ==================== ЗАПУСК ПРОГРАММЫ ====================
if __name__ == "__main__":
    try:
        system = EnhancedStockAnalysisSystem()
        system.run_continuous_analysis(interval_minutes=3)
    except Exception as e:
        logger.critical(f"💥 Критическая ошибка при запуске системы: {e}")
        sys.exit(1)