# ==================== УЛУЧШЕННЫЙ КЛАСС ИИ МОДЕЛИ С ПРОГНОЗОМ ПО СЕКТОРАМ ====================
import os
import tempfile
import traceback
from datetime import datetime
from urllib.request import urlretrieve
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

import library as l


class EnhancedStockAIModel:
    def __init__(self, model_path='enhanced_stock_ai_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.probabilistic_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = []
        self.model_metadata = {}
        self.sector_models = {}  # Отдельные модели для каждого сектора
        self.sector_scalers = {}  # Отдельные скейлеры для каждого сектора
        self.current_sector_prices = {}  # Текущие средние цены по секторам
        self.predicted_sector_prices = {}  # Предсказанные средние цены по секторам

        # Новая логика инициализации модели
        self._initialize_model_with_fallback()

    def _initialize_model_with_fallback(self):
        """Улучшенная инициализация модели с системой fallback"""
        print("🔄 Инициализация ИИ моделей...")

        # Стратегия 1: Попытка загрузить существующую модель
        if self._try_load_existing_model():
            print("✅ Модель загружена из файла")
            return

        # Стратегия 2: Попытка загрузить претренированную модель
        if self._try_download_pretrained_model():
            print("✅ Загружена претренированная модель")
            return

        # Стратегия 3: Создание новой модели
        print("📝 Создание новой модели...")
        self._create_new_models()
        print("✅ Создана новая модель")

    def _try_load_existing_model(self):
        """Попытка загрузки существующей модели"""
        try:
            if not os.path.exists(self.model_path):
                print("📝 Файл модели не найден")
                return False

            print(f"🔄 Загрузка модели из {self.model_path}")
            model_data = joblib.load(self.model_path)

            if isinstance(model_data, dict):
                return self._load_from_dict(model_data)
            else:
                return self._load_old_format(model_data)

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False

    def _try_download_pretrained_model(self):
        """Попытка загрузки претренированной модели"""
        try:
            # Здесь можно указать URL для загрузки претренированных моделей
            pretrained_url = l.PRETRAINED_MODEL_URLS.get('base_model')

            if not pretrained_url:
                print("ℹ️ URL претренированных моделей не указаны")
                return False

            print(f"🌐 Попытка загрузки претренированной модели...")

            # Создаем временный файл для загрузки
            with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp_file:
                temp_path = tmp_file.name

            try:
                # Загружаем модель
                urlretrieve(pretrained_url, temp_path)

                # Загружаем данные модели
                model_data = joblib.load(temp_path)

                if isinstance(model_data, dict):
                    success = self._load_from_dict(model_data)
                    if success:
                        # Сохраняем модель локально для будущего использования
                        self._save_models()
                        print("💾 Претренированная модель сохранена локально")
                    return success

            finally:
                # Удаляем временный файл
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

            return False

        except Exception as e:
            print(f"❌ Ошибка загрузки претренированной модели: {e}")
            return False

    def _load_from_dict(self, model_data):
        """Загрузка из словаря с проверкой всех компонентов"""
        try:
            required_keys = ['main_model', 'probabilistic_model', 'scaler']
            if not all(key in model_data for key in required_keys):
                print("❌ В файле модели отсутствуют необходимые компоненты")
                return False

            self.model = model_data['main_model']
            self.probabilistic_model = model_data['probabilistic_model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data.get('is_trained', True)
            self.training_history = model_data.get('training_history', [])
            self.model_metadata = model_data.get('model_metadata', {})

            # Загрузка моделей по секторам если они есть
            self.sector_models = model_data.get('sector_models', {})
            self.sector_scalers = model_data.get('sector_scalers', {})
            self.current_sector_prices = model_data.get('current_sector_prices', {})
            self.predicted_sector_prices = model_data.get('predicted_sector_prices', {})

            # Проверяем, что модели действительно инициализированы
            if self.model is None or self.probabilistic_model is None:
                print("❌ Загруженные модели равны None")
                return False

            print(f"📊 Модель загружена: {len(self.training_history)} тренировок, "
                  f"{self.model_metadata.get('total_training_samples', 0)} samples")
            return True

        except Exception as e:
            print(f"❌ Ошибка загрузки из словаря: {e}")
            return False

    def _load_old_format(self, model_data):
        """Загрузка старого формата модели"""
        try:
            if isinstance(model_data, (list, tuple)) and len(model_data) >= 2:
                self.model = model_data[0]
                self.probabilistic_model = model_data[1]
                self.is_trained = True
                return True
            return False
        except Exception as e:
            print(f"❌ Ошибка загрузки старого формата: {e}")
            return False

    def _create_new_models(self):
        """Создание новых моделей с гарантированной инициализацией"""
        try:
            # RandomForest с включенным warm_start для дообучения
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                warm_start=True  # Важно для дообучения!
            )

            # BayesianRidge для вероятностных прогнозов
            self.probabilistic_model = BayesianRidge(
                n_iter=200,
                tol=1e-3
            )

            # Инициализируем модели для каждого сектора
            self._initialize_sector_models()

            # Инициализируем scaler
            self.scaler = StandardScaler()

            self.is_trained = False
            self.model_metadata = {
                'creation_date': datetime.now().isoformat(),
                'total_training_samples': 0,
                'training_sessions': 0,
                'last_training': None,
                'model_type': 'EnhancedStockAIModel_v6',
                'supports_incremental_learning': True,
                'supports_sector_analysis': True
            }

            return True

        except Exception as e:
            print(f"❌ Критическая ошибка создания моделей: {e}")
            # Создаем простейшие модели в случае ошибки
            self.model = RandomForestRegressor(n_estimators=10, random_state=42)
            self.probabilistic_model = BayesianRidge()
            self._initialize_sector_models()
            self.is_trained = False
            return False

    def _initialize_sector_models(self):
        """Инициализация моделей для каждого сектора"""
        try:
            for sector in l.RUSSIAN_SECTORS.keys():
                # Создаем отдельную модель для каждого сектора
                self.sector_models[sector] = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=8,
                    random_state=42,
                    warm_start=True
                )
                self.sector_scalers[sector] = StandardScaler()

            print(f"✅ Инициализировано {len(self.sector_models)} моделей по секторам")
        except Exception as e:
            print(f"❌ Ошибка инициализации моделей по секторам: {e}")

    def train_models(self, prices_df, news_sentiment_by_date, incremental=False):
        """Улучшенное обучение моделей с поддержкой дообучения и обучением по секторам"""
        try:
            print("🔄 Запуск обучения моделей...")

            # Проверяем инициализацию моделей
            if self.model is None or self.probabilistic_model is None:
                print("⚠️ Модели не инициализированы, создаем новые...")
                self._create_new_models()

            # Подготавливаем данные для основной модели
            X, y = self.prepare_training_data(prices_df, news_sentiment_by_date)

            # Проверяем данные
            if not self._validate_training_data(X, y):
                print("❌ Невалидные данные для обучения")
                return False

            print(f"🎯 Обучение основной модели на {len(X)} примерах...")

            # Обработка данных с учетом дообучения
            try:
                if self.is_trained and incremental and hasattr(self.scaler, 'n_features_in_'):
                    # Для дообучения - partial_fit или transform
                    if hasattr(self.scaler, 'partial_fit'):
                        self.scaler.partial_fit(X)
                    X_processed = self.scaler.transform(X)
                else:
                    # Первое обучение или полное переобучение
                    X_processed = self.scaler.fit_transform(X)
            except Exception as scale_error:
                print(f"⚠️ Ошибка нормализации: {scale_error}, применяем fit_transform")
                X_processed = self.scaler.fit_transform(X)

            # ОБУЧЕНИЕ ОСНОВНОЙ МОДЕЛИ
            if self.is_trained and incremental:
                print("📚 Дообучение существующей модели...")
                if hasattr(self.model, 'warm_start') and self.model.warm_start:
                    current_estimators = self.model.n_estimators
                    self.model.n_estimators = current_estimators + 50
                    self.model.fit(X_processed, y)
                else:
                    self.model.fit(X_processed, y)

                self.probabilistic_model.fit(X_processed, y)
            else:
                # Первоначальное обучение
                print("🎓 Первоначальное обучение модели...")
                self.model.fit(X_processed, y)
                self.probabilistic_model.fit(X_processed, y)

            # ОБУЧЕНИЕ МОДЕЛЕЙ ПО СЕКТОРАМ
            print("🏢 Обучение моделей по секторам...")
            self._train_sector_models(prices_df, news_sentiment_by_date)

            # РАСЧЕТ ТЕКУЩИХ СРЕДНИХ ЦЕН ПО СЕКТОРАМ
            self._calculate_current_sector_prices(prices_df)

            # ПРОГНОЗ СРЕДНИХ ЦЕН ПО СЕКТОРАМ
            self._predict_sector_prices(prices_df, news_sentiment_by_date)

            # Обновляем статус и метаданные
            self.is_trained = True
            training_type = "incremental" if (self.is_trained and incremental) else "full"
            self._update_training_metadata(len(X), training_type)

            # Сохраняем модели
            self._save_models()

            print("✅ Обучение завершено успешно")
            return True

        except Exception as e:
            print(f"❌ Ошибка обучения моделей: {e}")
            traceback.print_exc()
            return False

    def _train_sector_models(self, prices_df, news_sentiment_by_date):
        """Обучение отдельных моделей для каждого сектора"""
        try:
            for sector, tickers in l.RUSSIAN_SECTORS.items():
                # Фильтруем данные только для тикеров этого сектора
                sector_tickers = [t for t in tickers if t in prices_df.columns]

                if len(sector_tickers) < 2:  # Нужно минимум 2 тикера для анализа
                    continue

                sector_prices = prices_df[sector_tickers]

                # Подготавливаем данные для сектора
                X_sector, y_sector = self.prepare_sector_training_data(
                    sector_prices, news_sentiment_by_date, sector
                )

                if len(X_sector) > 0 and len(y_sector) > 0:
                    # Обучаем модель сектора
                    X_processed = self.sector_scalers[sector].fit_transform(X_sector)
                    self.sector_models[sector].fit(X_processed, y_sector)
                    print(f"  ✅ Сектор '{sector}': обучено на {len(X_sector)} примерах")

        except Exception as e:
            print(f"⚠️ Ошибка обучения моделей по секторам: {e}")

    def _calculate_current_sector_prices(self, prices_df):
        """Расчет текущих средних цен по секторам"""
        try:
            self.current_sector_prices = {}

            for sector, tickers in l.RUSSIAN_SECTORS.items():
                # Берем только существующие тикеры
                valid_tickers = [t for t in tickers if t in prices_df.columns]

                if valid_tickers:
                    # Берем последние доступные цены
                    sector_prices = prices_df[valid_tickers].iloc[-1].dropna()

                    if len(sector_prices) > 0:
                        avg_price = sector_prices.mean()
                        self.current_sector_prices[sector] = {
                            'average_price': float(avg_price),
                            'ticker_count': len(sector_prices),
                            'min_price': float(sector_prices.min()),
                            'max_price': float(sector_prices.max()),
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        self.current_sector_prices[sector] = {
                            'average_price': 0.0,
                            'ticker_count': 0,
                            'min_price': 0.0,
                            'max_price': 0.0,
                            'timestamp': datetime.now().isoformat()
                        }

            print(f"📊 Рассчитаны текущие цены для {len(self.current_sector_prices)} секторов")

        except Exception as e:
            print(f"❌ Ошибка расчета текущих цен по секторам: {e}")

    def _predict_sector_prices(self, prices_df, news_sentiment_by_date):
        """Прогноз средних цен по секторам на следующую итерацию"""
        try:
            self.predicted_sector_prices = {}

            for sector, tickers in l.RUSSIAN_SECTORS.items():
                if sector not in self.sector_models:
                    continue

                # Создаем фичи для прогноза по сектору
                sector_features = self._create_sector_features(
                    sector, prices_df, news_sentiment_by_date
                )

                if sector_features and len(sector_features) > 0:
                    # Прогнозируем изменение цены
                    features_array = np.array(list(sector_features.values())).reshape(1, -1)
                    features_processed = self.sector_scalers[sector].transform(features_array)

                    predicted_change = self.sector_models[sector].predict(features_processed)[0]

                    # Рассчитываем предсказанную цену
                    current_avg = self.current_sector_prices.get(sector, {}).get('average_price', 0)
                    predicted_price = current_avg * (1 + predicted_change)

                    self.predicted_sector_prices[sector] = {
                        'predicted_price': float(predicted_price),
                        'predicted_change': float(predicted_change),
                        'current_price': float(current_avg),
                        'confidence': 0.8,  # Можно рассчитать на основе модели
                        'timestamp': datetime.now().isoformat()
                    }

            print(f"🔮 Сгенерированы прогнозы для {len(self.predicted_sector_prices)} секторов")

        except Exception as e:
            print(f"❌ Ошибка прогнозирования цен по секторам: {e}")

    def _create_sector_features(self, sector, prices_df, news_sentiment_by_date):
        """Создание признаков для прогноза по сектору"""
        try:
            features = {}
            tickers = l.RUSSIAN_SECTORS[sector]
            valid_tickers = [t for t in tickers if t in prices_df.columns]

            if not valid_tickers:
                return None

            sector_prices = prices_df[valid_tickers]

            # Базовые статистики по сектору
            current_prices = sector_prices.iloc[-1].dropna()
            if len(current_prices) == 0:
                return None

            returns = sector_prices.pct_change().dropna()

            features['sector_volatility'] = returns.std().mean() if not returns.empty else 0.02
            features['sector_momentum'] = (sector_prices.iloc[-1] / sector_prices.iloc[-5] - 1).mean() if len(
                sector_prices) >= 5 else 0
            features['sector_volume'] = len(valid_tickers) / len(tickers)  # Доля доступных тикеров

            # Сентимент новостей по сектору
            current_sentiment = {}
            if news_sentiment_by_date:
                latest_date = max(news_sentiment_by_date.keys())
                current_sentiment = news_sentiment_by_date[latest_date]

            features['sector_sentiment'] = current_sentiment.get(sector, 0)

            # Общие рыночные тренды
            features['market_trend'] = prices_df.mean(axis=1).pct_change().iloc[-1] if len(prices_df) > 1 else 0

            return features

        except Exception as e:
            print(f"⚠️ Ошибка создания признаков для сектора {sector}: {e}")
            return None

    def prepare_sector_training_data(self, sector_prices, news_sentiment_by_date, sector):
        """Подготовка данных для обучения моделей секторов"""
        X, y = [], []

        if sector_prices.empty:
            return np.array([]), np.array([])

        try:
            # Используем исторические данные для создания примеров
            for i in range(10, len(sector_prices)):
                # Историческое окно для фичей
                historical_window = sector_prices.iloc[i - 10:i]

                if historical_window.empty:
                    continue

                # Создаем фичи из исторического окна
                features = self._create_historical_sector_features(
                    historical_window, news_sentiment_by_date, sector, i
                )

                if features:
                    # Целевая переменная - изменение цены в следующем периоде
                    if i < len(sector_prices) - 1:
                        current_avg = historical_window.iloc[-1].mean()
                        next_avg = sector_prices.iloc[i + 1].mean()
                        price_change = (next_avg - current_avg) / current_avg if current_avg != 0 else 0

                        X.append(list(features.values()))
                        y.append(price_change)

            return np.array(X), np.array(y)

        except Exception as e:
            print(f"❌ Ошибка подготовки данных для сектора {sector}: {e}")
            return np.array([]), np.array([])

    def _create_historical_sector_features(self, historical_prices, news_sentiment_by_date, sector, index):
        """Создание исторических признаков для сектора"""
        try:
            features = {}

            # Статистики цен
            current_prices = historical_prices.iloc[-1].dropna()
            if len(current_prices) == 0:
                return None

            returns = historical_prices.pct_change().dropna()

            features['hist_volatility'] = returns.std().mean() if not returns.empty else 0.02
            features['price_momentum'] = (historical_prices.iloc[-1] / historical_prices.iloc[-5] - 1).mean() if len(
                historical_prices) >= 5 else 0
            features['price_trend'] = (historical_prices.iloc[-1] / historical_prices.iloc[0] - 1).mean()

            # Сентимент на соответствующую дату
            sentiment_date = historical_prices.index[-1].date() if hasattr(historical_prices.index[-1], 'date') else \
            historical_prices.index[-1]
            sector_sentiment = 0

            for date, sentiment_dict in news_sentiment_by_date.items():
                if date == sentiment_date:
                    sector_sentiment = sentiment_dict.get(sector, 0)
                    break

            features['hist_sentiment'] = sector_sentiment

            return features

        except Exception as e:
            print(f"⚠️ Ошибка создания исторических признаков: {e}")
            return None

    def get_sector_analysis(self):
        """Получение полного анализа по секторам"""
        return {
            'current_sector_prices': self.current_sector_prices,
            'predicted_sector_prices': self.predicted_sector_prices,
            'analysis_timestamp': datetime.now().isoformat(),
            'sectors_analyzed': list(self.current_sector_prices.keys())
        }

    def _update_training_metadata(self, new_samples_count, training_type):
        """Обновление метаданных обучения"""
        current_total = self.model_metadata.get('total_training_samples', 0)
        self.model_metadata['total_training_samples'] = current_total + new_samples_count
        self.model_metadata['training_sessions'] = self.model_metadata.get('training_sessions', 0) + 1
        self.model_metadata['last_training'] = datetime.now().isoformat()
        self.model_metadata['last_training_type'] = training_type
        self.model_metadata['sector_models_count'] = len(self.sector_models)

        # Сохраняем историю тренировок
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'samples': new_samples_count,
            'total_samples': self.model_metadata['total_training_samples'],
            'type': training_type,
            'sectors_analyzed': len(self.current_sector_prices)
        })

    @staticmethod
    def _validate_training_data(X, y):
        """Проверка валидности данных для обучения"""
        if X is None or y is None:
            print("❌ Данные для обучения равны None")
            return False

        if len(X) == 0 or len(y) == 0:
            print("❌ Пустые данные для обучения")
            return False

        if len(X) != len(y):
            print("❌ Несовпадение размеров X и y")
            return False

        return True

    def _save_models(self):
        """Сохранение моделей с проверкой"""
        try:
            if self.model is None or self.probabilistic_model is None:
                print("⚠️ Нечего сохранять: модели не инициализированы")
                return False

            models_to_save = {
                'main_model': self.model,
                'probabilistic_model': self.probabilistic_model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'model_metadata': self.model_metadata,
                'sector_models': self.sector_models,
                'sector_scalers': self.sector_scalers,
                'current_sector_prices': self.current_sector_prices,
                'predicted_sector_prices': self.predicted_sector_prices
            }

            joblib.dump(models_to_save, self.model_path)
            print(f"💾 Модели успешно сохранены в {self.model_path}")
            return True

        except Exception as e:
            print(f"❌ Ошибка сохранения моделей: {e}")
            return False

    def create_advanced_features(self, prices_df, news_sentiment):
        """Создание расширенных признаков"""
        features = {}

        if prices_df.empty:
            print("⚠️ Нет данных для создания признаков")
            return features

        for ticker in prices_df.columns:
            try:
                if ticker not in prices_df:
                    continue

                price_series = prices_df[ticker].dropna()
                if len(price_series) < 30:
                    continue

                # Упрощенный расчет признаков для демонстрации
                returns = price_series.pct_change().dropna()
                if len(returns) < 15:
                    continue

                # Базовые признаки
                volatility = returns.std()
                current_price = price_series.iloc[-1]
                ma_short = price_series.rolling(window=5).mean().iloc[-1]
                ma_long = price_series.rolling(window=20).mean().iloc[-1]

                features[ticker] = {
                    'volatility': float(volatility) if not pd.isna(volatility) else 0.02,
                    'price_ratio': float(current_price / ma_long) if ma_long != 0 else 1.0,
                    'ma_ratio': float(ma_short / ma_long) if ma_long != 0 else 1.0,
                    'sector_sentiment': float(news_sentiment.get(self.get_sector_for_ticker(ticker), 0))
                }

            except Exception as e:
                print(e)  # TODO: define
                continue

        return features

    @staticmethod
    def get_sector_for_ticker(ticker):
        """Определение сектора для тикера"""
        for sector, tickers in l.RUSSIAN_SECTORS.items():
            if ticker in tickers:
                return sector
        return 'Общие'

    def prepare_training_data(self, prices_df, news_sentiment_by_date, forecast_days=5):
        """Подготовка данных для обучения"""
        X, y = [], []

        if prices_df.empty:
            return np.array([]), np.array([])

        # Упрощенная логика подготовки данных
        current_sentiment = {}
        if news_sentiment_by_date:
            latest_date = max(news_sentiment_by_date.keys())
            current_sentiment = news_sentiment_by_date[latest_date]

        features_dict = self.create_advanced_features(prices_df, current_sentiment)

        for ticker, features in features_dict.items():
            if features and len(features) > 0:
                feature_vector = list(features.values())
                # Простая целевая переменная для демонстрации
                target = np.random.uniform(-0.1, 0.1)  # Замените реальной логикой
                X.append(feature_vector)
                y.append(target)

        if len(X) > 0:
            return np.array(X), np.array(y)
        else:
            return np.array([]), np.array([])

    def predict_with_confidence(self, current_features):
        """Прогноз с оценкой достоверности"""
        if not self.is_trained or self.model is None:
            return 0.0, 0.0

        try:
            if isinstance(current_features, dict):
                feature_vector = list(current_features.values())
            else:
                feature_vector = current_features

            if len(feature_vector) == 0:
                return 0.0, 0.0

            features_array = np.array(feature_vector).reshape(1, -1)
            features_array = self.scaler.transform(features_array)

            main_pred = self.model.predict(features_array)[0]
            probabilistic_pred, probabilistic_std = self.probabilistic_model.predict(features_array, return_std=True)

            combined_pred = 0.7 * main_pred + 0.3 * probabilistic_pred
            confidence = max(0.0, min(1.0, 1.0 - probabilistic_std[0]))

            return float(combined_pred), float(confidence)

        except Exception as e:
            print(f"❌ Ошибка прогнозирования: {e}")
            return 0.0, 0.0

    def get_training_info(self):
        """Получение информации о тренировке модели"""
        if not self.training_history:
            return "Модель еще не обучалась"

        latest = self.training_history[-1]
        return {
            'last_training': latest['timestamp'],
            'samples': latest['total_samples'],
            'total_trainings': len(self.training_history),
            'model_metadata': self.model_metadata,
            'sector_analysis_available': len(self.current_sector_prices) > 0
        }

    def get_model_status(self):
        """Получение статуса модели"""
        return {
            'is_trained': self.is_trained,
            'model_initialized': self.model is not None,
            'training_sessions': len(self.training_history),
            'total_samples': self.model_metadata.get('total_training_samples', 0),
            'supports_incremental': self.model_metadata.get('supports_incremental_learning', False),
            'sector_models_count': len(self.sector_models),
            'current_sector_data': len(self.current_sector_prices),
            'predicted_sector_data': len(self.predicted_sector_prices)
        }

  #  print(f"🤖новый анализ: {is_trained}")