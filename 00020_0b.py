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
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import re
import threading
from flask import Flask, render_template_string, request
import json
import traceback
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# ==================== КОНФИГУРАЦИЯ ====================
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


# ==================== КЛАСС ДЛЯ УЧЕТА БРОКЕРСКИХ КОМИССИЙ ====================
class BrokerCommissionManager:
    def __init__(self, config_file='broker_config.json'):
        self.config_file = config_file
        self.commission_config = self._load_commission_config()

    def _load_commission_config(self):
        """Загрузка конфигурации комиссий из файла"""
        default_config = {
            'commission_percent': 0.05,  # 0.05% от оборота
            'min_commission_per_trade': 1.0,  # Минимальная комиссия за сделку в рублях
            'min_commission_per_day': 10.0,  # Минимальная комиссия за день в рублях
            'tax_percent': 0.13,  # Налог на прибыль 13%
            'min_trade_amount': 1000,  # Минимальная сумма сделки
            'exchange_fee_percent': 0.01,  # Биржевой сбор
            'clearing_fee_percent': 0.01  # Клиринговый сбор
        }

        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # Объединяем с конфигом по умолчанию
                    config = {**default_config, **user_config}
                print(f"✅ Конфигурация комиссий загружена из {self.config_file}")
            else:
                config = default_config
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                print(f"📁 Создан файл конфигурации комиссий {self.config_file}")

            return config

        except Exception as e:
            print(f"❌ Ошибка загрузки конфигурации комиссий: {e}")
            return default_config

    def calculate_trade_commission(self, trade_amount, is_intraday=False):
        """Расчет комиссии за сделку"""
        commission = trade_amount * (self.commission_config['commission_percent'] / 100)
        commission = max(commission, self.commission_config['min_commission_per_trade'])

        # Добавляем биржевые сборы
        exchange_fee = trade_amount * (self.commission_config['exchange_fee_percent'] / 100)
        clearing_fee = trade_amount * (self.commission_config['clearing_fee_percent'] / 100)

        total_commission = commission + exchange_fee + clearing_fee

        return {
            'broker_commission': commission,
            'exchange_fee': exchange_fee,
            'clearing_fee': clearing_fee,
            'total_commission': total_commission,
            'trade_amount': trade_amount
        }

    def calculate_daily_commission(self, daily_trades):
        """Расчет комиссий за день с учетом минимальной суммы"""
        daily_total = sum(trade['total_commission'] for trade in daily_trades)
        min_daily = self.commission_config['min_commission_per_day']

        if daily_total < min_daily:
            additional_commission = min_daily - daily_total
            return daily_total + additional_commission, additional_commission
        else:
            return daily_total, 0

    def calculate_tax(self, realized_profit):
        """Расчет налога на прибыль"""
        if realized_profit > 0:
            return realized_profit * self.commission_config['tax_percent']
        return 0

    def update_config(self, new_config):
        """Обновление конфигурации комиссий"""
        try:
            self.commission_config = {**self.commission_config, **new_config}
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.commission_config, f, ensure_ascii=False, indent=2)
            print("✅ Конфигурация комиссий обновлена")
            return True
        except Exception as e:
            print(f"❌ Ошибка обновления конфигурации: {e}")
            return False


def load_russian_feeds_from_json(file_path='russian_feeds.json'):
    """Загрузка RSS-лент из JSON файла"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                feeds = json.load(f)
            print(f"✅ RSS-ленты загружены из {file_path}")
            return feeds
        else:
            default_feeds = {
                "Ведомости": "https://www.vedomosti.ru/rss/news",
                "Коммерсант": "https://www.kommersant.ru/RSS/news.xml",
                "Интерфакс": "https://www.interfax.ru/rss.asp",
                "ТАСС": "https://tass.ru/rss/v2.xml",
                "Лента.ру": "https://lenta.ru/rss/news",
                "Финам": "https://www.finam.ru/analysis/conews/rsspoint/",
                "БФМ": "https://www.bfm.ru/news.rss?rubric=19",
                "ФинМаркет": "https://www.finmarket.ru/rss/mainnews.asp"
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(default_feeds, f, ensure_ascii=False, indent=2)
            print(f"📁 Создан файл {file_path} с RSS-лентами по умолчанию")
            return default_feeds
    except Exception as e:
        print(f"❌ Ошибка загрузки RSS-лент: {e}")
        return {}


# Загружаем RSS-ленты из JSON файла
RUSSIAN_FEEDS = load_russian_feeds_from_json()


# ==================== КЛАСС ДЛЯ СБОРА НОВОСТЕЙ ====================
class NewsCollector:
    def __init__(self, feeds_config=None):
        self.feeds = feeds_config if feeds_config is not None else RUSSIAN_FEEDS
        self.feeds_file = 'russian_feeds.json'
        self._ensure_feeds_file()

    def _ensure_feeds_file(self):
        """Проверяет наличие файла с RSS-лентами"""
        if not os.path.exists(self.feeds_file):
            print(f"📁 Создаем файл {self.feeds_file} с RSS-лентами...")
            self.save_feeds_to_file()

    def save_feeds_to_file(self):
        """Сохраняет текущие RSS-ленты в файл"""
        try:
            with open(self.feeds_file, 'w', encoding='utf-8') as f:
                json.dump(self.feeds, f, ensure_ascii=False, indent=2)
            print(f"💾 RSS-ленты сохранены в {self.feeds_file}")
        except Exception as e:
            print(f"❌ Ошибка сохранения RSS-лент: {e}")

    def fetch_all_news(self, max_items_per_feed=15):
        """Сбор новостей из всех RSS-лент"""
        all_news = []

        for source, url in self.feeds.items():
            try:
                print(f"📰 Загрузка новостей из {source}...")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                feed = feedparser.parse(url)
                time.sleep(0.3)

                if not feed.entries:
                    print(f"  ⚠️ Нет новостей в {source}")
                    continue

                items_added = 0
                for entry in feed.entries[:max_items_per_feed]:
                    try:
                        title = getattr(entry, 'title', '').strip()
                        if not title:
                            continue

                        news_item = {
                            'source': source,
                            'title': title,
                            'link': getattr(entry, 'link', ''),
                            'published': getattr(entry, 'published', ''),
                            'summary': getattr(entry, 'summary', '')[:200],
                            'timestamp': datetime.now()
                        }

                        all_news.append(news_item)
                        items_added += 1

                    except Exception:
                        continue

                if items_added > 0:
                    print(f"  ✅ Добавлено {items_added} новостей")
                else:
                    print(f"  ⚠️ Не удалось получить новости из {source}")

            except Exception as e:
                print(f"❌ Ошибка при загрузке из {source}: {e}")
                continue

        news_df = pd.DataFrame(all_news)
        if not news_df.empty:
            news_df = news_df.drop_duplicates(subset=['title']).reset_index(drop=True)

        print(f"📊 Всего собрано новостей: {len(news_df)}")
        return news_df


# ==================== ИСПРАВЛЕННЫЙ КЛАСС ДЛЯ ДАННЫХ АКЦИЙ ====================
class CorrectedStockDataCollector:
    def __init__(self):
        self.base_url = "https://iss.moex.com/iss"

    def get_current_prices(self, tickers):
        """Получение ТЕКУЩИХ цен акций для портфеля"""
        current_prices = {}

        for ticker in tickers:
            try:
                url = f"{self.base_url}/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"

                params = {
                    'iss.meta': 'off',
                    'iss.only': 'marketdata',
                    'marketdata.columns': 'SECID,LAST'
                }

                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    marketdata = data.get('marketdata', {})

                    if marketdata.get('data') and marketdata['data']:
                        last_price = marketdata['data'][0][1]
                        if last_price is not None:
                            current_prices[ticker] = float(last_price)
                            print(f"  ✅ {ticker}: {last_price} ₽ (текущая цена)")
                        else:
                            prev_price = self._get_previous_close(ticker)
                            if prev_price:
                                current_prices[ticker] = prev_price
                                print(f"  ⚠️ {ticker}: {prev_price} ₽ (цена закрытия)")
                            else:
                                print(f"  ❌ {ticker}: цена недоступна")
                    else:
                        print(f"  ❌ {ticker}: нет данных marketdata")
                else:
                    print(f"  ❌ {ticker}: ошибка HTTP {response.status_code}")

                time.sleep(0.3)

            except Exception as e:
                print(f"  ❌ Ошибка получения цены {ticker}: {e}")
                continue

        return current_prices

    def _get_previous_close(self, ticker):
        """Получение цены закрытия предыдущего дня"""
        try:
            url = f"{self.base_url}/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
            params = {
                'iss.meta': 'off',
                'iss.only': 'securities',
                'securities.columns': 'PREVLEGALCLOSEPRICE'
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                securities = data.get('securities', {})
                if securities.get('data'):
                    prev_price = securities['data'][0][0]
                    return float(prev_price) if prev_price else None
        except:
            pass
        return None

    def get_historical_data(self, tickers, days=60):
        """Получение исторических данных для обучения ИИ"""
        stock_data = {}
        successful_downloads = 0

        for ticker in tickers:
            try:
                url = f"{self.base_url}/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json"
                from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

                params = {
                    'from': from_date,
                    'till': datetime.now().strftime('%Y-%m-%d'),
                    'interval': 24
                }

                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    candles = data.get('candles', {}).get('data', [])

                    if candles:
                        dates = []
                        prices = []

                        for candle in candles:
                            close_price = candle[1]
                            date_str = candle[6]

                            if date_str and close_price:
                                dates.append(date_str)
                                prices.append(float(close_price))

                        if len(prices) >= 10:
                            series = pd.Series(prices, index=dates)
                            stock_data[ticker] = series
                            successful_downloads += 1

                time.sleep(0.2)

            except Exception as e:
                print(f"⚠️ Ошибка загрузки исторических данных {ticker}: {e}")
                continue

        print(f"📈 Успешно загружено исторических данных для {successful_downloads} тикеров")

        if stock_data:
            prices_df = pd.DataFrame(stock_data)
            prices_df = prices_df.ffill().bfill()
            prices_df = prices_df.dropna(axis=1, thresh=len(prices_df) * 0.7)
            return prices_df
        else:
            print("🔄 Создание тестовых исторических данных...")
            return self.create_test_data(tickers, days)

    def create_test_data(self, tickers, days):
        """Создание реалистичных тестовых данных"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        base_prices = {
            'SBER': 250, 'GAZP': 160, 'LKOH': 5800, 'GMKN': 25000, 'NLMK': 180,
            'MTSS': 270, 'ROSN': 420, 'TATN': 380, 'VTBR': 0.03, 'MOEX': 150,
            'ALRS': 450, 'CHMK': 50, 'MTLR': 700, 'TRNFP': 150000,
            'FIVE': 2100, 'DSKY': 600, 'LNTA': 2000, 'OZON': 2200,
            'POLY': 700, 'RGSS': 300, 'YNDX': 2900,
            'GCHE': 1500, 'UPRO': 20, 'WUSH': 5,
            'RTKM': 70, 'MGNT': 4500,
            'PHOR': 8000, 'AKRN': 6500, 'ODVA': 200,
            'FEES': 20, 'HYDR': 1, 'IRAO': 3,
            'PIKK': 1100, 'LSRG': 800, 'UNAC': 500,
            'AFLT': 40, 'NMTP': 300, 'TRMK': 1500
        }

        prices_df = pd.DataFrame(index=dates)
        np.random.seed(42)

        for ticker in tickers:
            base_price = base_prices.get(ticker, 100)
            trend = np.random.choice([-0.001, 0, 0.001, 0.002])
            volatility = 0.02 + np.random.random() * 0.03
            returns = np.random.normal(trend, volatility, days)

            prices = [base_price]
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 0.01))

            prices_df[ticker] = prices

        print(f"✅ Созданы тестовые исторические данные для {len(tickers)} тикеров")
        return prices_df


# ==================== КЛАСС АНАЛИЗА НОВОСТЕЙ ====================
class NewsAnalyzer:
    def __init__(self):
        self.sector_keywords = self._build_sector_keywords()
        self.sentiment_words = self._build_sentiment_lexicon()

    def _build_sector_keywords(self):
        return {
            'Энергетика': ['нефть', 'газ', 'нефтегаз', 'энергетик', 'нефтяной', 'газовый', 'нефтедобыча'],
            'Индустриальный': ['промышленност', 'завод', 'производств', 'индустриальн', 'оборудован'],
            'Базовые ресурсы и материалы': ['металл', 'сталь', 'никель', 'алюмин', 'медь', 'руд', 'горнодобывающ'],
            'Розничная и оптовая торговля': ['ритейл', 'магазин', 'торговля', 'розничн', 'покуп', 'продаж', 'товар'],
            'Медицина, фармацевтика, охрана здоровья': ['фарма', 'медицин', 'лекарств', 'препарат', 'витамин',
                                                        'здоровье'],
            'Потребительский': ['потребитель', 'спрос', 'розничн', 'покуп', 'товар'],
            'Телекоммуникации': ['связь', 'телеком', 'интернет', 'мобильн', 'тариф', 'абонент'],
            'Химия и нефтехимия': ['химия', 'удобрен', 'нефтехим', 'полимер', 'пластик'],
            'Электроэнергетика': ['электроэнерг', 'энергосбыт', 'электросет', 'электростанц'],
            'Строительство': ['строитель', 'девелопер', 'недвиж', 'жилье', 'квартир'],
            'Транспорт': ['авиа', 'транспорт', 'порт', 'аэропорт', 'груз', 'логистик']
        }

    def _build_sentiment_lexicon(self):
        return {
            'positive': ['рост', 'вырос', 'увеличил', 'прибыль', 'доход', 'успех', 'рекорд', 'улучшение', 'позитив'],
            'negative': ['падение', 'снижен', 'упал', 'сокращен', 'убыток', 'проблем', 'кризис', 'слаб', 'негатив'],
            'intensifiers': ['значительн', 'резк', 'существенн', 'кардинальн', 'масштабн']
        }

    def analyze_sentiment(self, text):
        """Анализ тональности текста"""
        if not isinstance(text, str):
            return 0.0

        text_lower = text.lower()
        positive_score = 0
        negative_score = 0

        for word in self.sentiment_words['positive']:
            if word in text_lower:
                positive_score += 1
                for intensifier in self.sentiment_words['intensifiers']:
                    if f"{intensifier} {word}" in text_lower:
                        positive_score += 0.5
                        break

        for word in self.sentiment_words['negative']:
            if word in text_lower:
                negative_score += 1
                for intensifier in self.sentiment_words['intensifiers']:
                    if f"{intensifier} {word}" in text_lower:
                        negative_score += 0.5
                        break

        total_score = positive_score - negative_score
        return max(-1.0, min(1.0, total_score * 0.2))

    def identify_sectors(self, text):
        """Идентификация секторов из текста"""
        if not isinstance(text, str):
            return []

        text_lower = text.lower()
        sectors_found = []

        for sector, keywords in self.sector_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    sectors_found.append(sector)
                    break

        return sectors_found if sectors_found else ['Общие']

    def predict_sector_sentiment(self, news_df):
        """Прогнозирование сентимента по секторам"""
        sector_sentiments = {sector: [] for sector in RUSSIAN_SECTORS.keys()}

        for _, news in news_df.iterrows():
            title = news['title']
            summary = news.get('summary', '')
            text = f"{title}. {summary}"

            sentiment = self.analyze_sentiment(text)
            sectors = self.identify_sectors(text)

            for sector in sectors:
                if sector in sector_sentiments:
                    sector_sentiments[sector].append(sentiment)

        recommendations = {}
        for sector, sentiments in sector_sentiments.items():
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                recommendation = int(round(avg_sentiment * 5))
                recommendation = max(-5, min(5, recommendation))
            else:
                recommendation = 0

            recommendations[sector] = recommendation

        return recommendations


# ==================== ИСПРАВЛЕННЫЙ КЛАСС ИИ МОДЕЛИ С УЧЕТОМ КОМИССИЙ ====================
class CommissionAwareStockAIModel:
    def __init__(self, model_path='commission_aware_stock_ai_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.probabilistic_model = None
        self.advanced_model = None
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.is_trained = False
        self.training_history = []
        self.model_metadata = {}
        self.commission_manager = BrokerCommissionManager()
        self._initialize_model_with_fallback()

    def _initialize_model_with_fallback(self):
        """Инициализация модели с системой fallback"""
        print("🔄 Инициализация ИИ моделей с учетом комиссий...")

        if self._try_load_existing_model():
            print("✅ Модель загружена из файла")
            return

        print("📝 Создание новой модели...")
        self._create_new_models()
        print("✅ Создана новая модель с учетом комиссий")

    def _try_load_existing_model(self):
        """Попытка загрузки существующей модели"""
        try:
            if not os.path.exists(self.model_path):
                return False

            model_data = joblib.load(self.model_path)
            return self._load_from_dict(model_data)

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False

    def _load_from_dict(self, model_data):
        """Загрузка из словаря"""
        try:
            required_keys = ['main_model', 'probabilistic_model', 'advanced_model', 'scaler', 'imputer']
            if not all(key in model_data for key in required_keys):
                return False

            self.model = model_data['main_model']
            self.probabilistic_model = model_data['probabilistic_model']
            self.advanced_model = model_data['advanced_model']
            self.scaler = model_data['scaler']
            self.imputer = model_data['imputer']
            self.is_trained = model_data.get('is_trained', True)
            self.training_history = model_data.get('training_history', [])
            self.model_metadata = model_data.get('model_metadata', {})

            return True

        except Exception as e:
            print(f"❌ Ошибка загрузки из словаря: {e}")
            return False

    def _create_new_models(self):
        """Создание новых моделей"""
        try:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                warm_start=True
            )

            self.probabilistic_model = BayesianRidge(n_iter=200, tol=1e-3)

            self.advanced_model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )

            self.imputer = SimpleImputer(strategy='median')
            self.scaler = RobustScaler()

            self.is_trained = False
            self.model_metadata = {
                'creation_date': datetime.now().isoformat(),
                'total_training_samples': 0,
                'training_sessions': 0,
                'last_training': None,
                'model_type': 'CommissionAwareStockAIModel_v1'
            }

        except Exception as e:
            print(f"❌ Ошибка создания моделей: {e}")
            self.model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.probabilistic_model = BayesianRidge()
            self.advanced_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
            self.imputer = SimpleImputer(strategy='median')
            self.scaler = RobustScaler()
            self.is_trained = False

    def calculate_technical_indicators(self, price_series):
        """Расчет технических индикаторов с обработкой ошибок"""
        if len(price_series) < 20:
            return {}

        try:
            returns = price_series.pct_change().dropna()
            if len(returns) < 5:
                return {}

            sma_5 = price_series.rolling(5).mean().iloc[-1] if len(price_series) >= 5 else price_series.iloc[-1]
            sma_20 = price_series.rolling(20).mean().iloc[-1] if len(price_series) >= 20 else price_series.iloc[-1]
            sma_50 = price_series.rolling(50).mean().iloc[-1] if len(price_series) >= 50 else sma_20

            delta = price_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if not rs.empty and not np.isnan(rs.iloc[-1]) else 50

            volatility_short = returns.tail(10).std() if len(returns) >= 10 else 0.02
            volatility_long = returns.std() if len(returns) >= 20 else 0.02

            momentum_1m = (price_series.iloc[-1] / price_series.iloc[-min(22, len(price_series))] - 1)
            momentum_3m = (price_series.iloc[-1] / price_series.iloc[-min(66, len(price_series))] - 1)

            volatility_ratio = volatility_short / volatility_long if volatility_long > 0 else 1.0
            momentum_ratio = momentum_1m / momentum_3m if momentum_3m != 0 else 1.0
            sma_ratio_5_20 = sma_5 / sma_20 if sma_20 != 0 else 1.0
            sma_ratio_20_50 = sma_20 / sma_50 if sma_50 != 0 else 1.0
            price_vs_sma20 = price_series.iloc[-1] / sma_20 if sma_20 != 0 else 1.0

            return {
                'sma_ratio_5_20': float(sma_ratio_5_20),
                'sma_ratio_20_50': float(sma_ratio_20_50),
                'price_vs_sma20': float(price_vs_sma20),
                'rsi': float(rsi),
                'volatility_ratio': float(volatility_ratio),
                'momentum_1m': float(momentum_1m),
                'momentum_3m': float(momentum_3m),
                'momentum_ratio': float(momentum_ratio),
                'volatility': float(volatility_long),
                'current_price': float(price_series.iloc[-1])
            }

        except Exception as e:
            print(f"⚠️ Ошибка расчета индикаторов: {e}")
            return {}

    def create_advanced_features(self, prices_df, news_sentiment):
        """Создание расширенных признаков с обработкой ошибок"""
        features = {}

        if prices_df.empty:
            return features

        for ticker in prices_df.columns:
            try:
                if ticker not in prices_df:
                    continue

                price_series = prices_df[ticker].dropna()
                if len(price_series) < 20:
                    continue

                technical_features = self.calculate_technical_indicators(price_series)
                if not technical_features:
                    continue

                sector = self.get_sector_for_ticker(ticker)
                sector_sentiment = news_sentiment.get(sector, 0)

                features[ticker] = {
                    **technical_features,
                    'sector_sentiment': float(sector_sentiment)
                }

            except Exception as e:
                print(f"⚠️ Ошибка создания признаков для {ticker}: {e}")
                continue

        return self._clean_features(features)

    def _clean_features(self, features_dict):
        """Очистка признаков от NaN и бесконечных значений"""
        cleaned_features = {}
        for ticker, features in features_dict.items():
            try:
                feature_array = np.array(list(features.values()), dtype=np.float64)
                feature_array = np.nan_to_num(feature_array, nan=np.nan, posinf=1e6, neginf=-1e6)

                if not np.any(np.isnan(feature_array)):
                    cleaned_features[ticker] = features
                else:
                    print(f"⚠️ Пропущен {ticker}: содержатся NaN значения")

            except (ValueError, TypeError) as e:
                print(f"⚠️ Пропущен {ticker}: ошибка преобразования - {e}")
                continue

        return cleaned_features

    def get_sector_for_ticker(self, ticker):
        """Определение сектора для тикера"""
        for sector, tickers in RUSSIAN_SECTORS.items():
            if ticker in tickers:
                return sector
        return 'Общие'

    def prepare_training_data(self, prices_df, news_sentiment_by_date, forecast_days=10):
        """Подготовка данных для обучения с учетом комиссий"""
        X_basic, X_advanced, y_basic, y_advanced = [], [], [], []

        if prices_df.empty:
            return np.array([]), np.array([]), np.array([]), np.array([])

        current_sentiment = {}
        if news_sentiment_by_date:
            latest_date = max(news_sentiment_by_date.keys())
            current_sentiment = news_sentiment_by_date[latest_date]

        features_dict = self.create_advanced_features(prices_df, current_sentiment)

        if not features_dict:
            print("❌ Нет валидных признаков для обучения")
            return np.array([]), np.array([]), np.array([]), np.array([])

        for ticker, features in features_dict.items():
            try:
                if features and len(features) > 0:
                    price_series = prices_df[ticker].dropna()
                    if len(price_series) > forecast_days:
                        current_idx = -1
                        future_idx = -forecast_days - 1

                        if len(price_series) > abs(future_idx):
                            current_price = price_series.iloc[current_idx]
                            future_price = price_series.iloc[future_idx]

                            # Базовая доходность без учета комиссий
                            raw_return = (current_price / future_price - 1)

                            # ✅ УЧЕТ КОМИССИЙ ПРИ ОБУЧЕНИИ
                            # Предполагаем одну сделку при покупке и одну при продаже
                            trade_amount = future_price * 1000  # Условная сумма сделки
                            commission_info = self.commission_manager.calculate_trade_commission(trade_amount)

                            # Учитываем комиссии в доходности
                            commission_impact = commission_info[
                                                    'total_commission'] / trade_amount * 2  # Покупка + продажа
                            adjusted_return = raw_return - commission_impact

                            feature_vector = list(features.values())

                            if not any(np.isnan(feature_vector)) and not np.isnan(adjusted_return):
                                X_basic.append(feature_vector)
                                X_advanced.append(feature_vector)
                                y_basic.append(adjusted_return)  # Используем скорректированную доходность
                                y_advanced.append(adjusted_return)

            except Exception as e:
                print(f"⚠️ Ошибка подготовки данных для {ticker}: {e}")
                continue

        print(f"📊 Подготовлено {len(X_basic)} валидных примеров для обучения с учетом комиссий")
        return (np.array(X_basic), np.array(X_advanced),
                np.array(y_basic), np.array(y_advanced))

    def _safe_fit_transform(self, X, incremental=False):
        """Безопасное преобразование данных с обработкой NaN"""
        if len(X) == 0:
            return X

        X_imputed = self.imputer.fit_transform(X) if not self.is_trained or not incremental else self.imputer.transform(
            X)
        if not self.is_trained or not incremental:
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_scaled = self.scaler.transform(X_imputed)

        return X_scaled

    def train_models(self, prices_df, news_sentiment_by_date, incremental=True):
        """Обучение моделей с обработкой ошибок"""
        try:
            print("🔄 Запуск обучения моделей с учетом комиссий...")

            if self.model is None:
                self._create_new_models()

            X_basic, X_advanced, y_basic, y_advanced = self.prepare_training_data(
                prices_df, news_sentiment_by_date
            )

            if len(X_basic) < 5:
                print(f"❌ Недостаточно данных для обучения: {len(X_basic)} примеров")
                return False

            print(f"🎯 Обучение на {len(X_basic)} примерах с учетом комиссий...")

            try:
                X_basic_processed = self._safe_fit_transform(X_basic, incremental)
                X_advanced_processed = self._safe_fit_transform(X_advanced, incremental)
            except Exception as e:
                print(f"❌ Ошибка обработки данных: {e}")
                return False

            try:
                if self.is_trained and incremental:
                    print("📚 Дообучение моделей...")
                    if hasattr(self.model, 'warm_start') and self.model.warm_start:
                        self.model.n_estimators += 30

                    self.model.fit(X_basic_processed, y_basic)
                    self.probabilistic_model.fit(X_basic_processed, y_basic)
                    self.advanced_model.fit(X_advanced_processed, y_advanced)
                else:
                    print("🎓 Первоначальное обучение моделей...")
                    self.model.fit(X_basic_processed, y_basic)
                    self.probabilistic_model.fit(X_basic_processed, y_basic)
                    self.advanced_model.fit(X_advanced_processed, y_advanced)

            except Exception as e:
                print(f"❌ Ошибка обучения моделей: {e}")
                return False

            self.is_trained = True
            self._update_training_metadata(len(X_basic))
            self._save_models()

            avg_return = np.mean(y_advanced) if len(y_advanced) > 0 else 0
            print(f"✅ Обучение завершено. Средняя доходность с учетом комиссий: {avg_return:.4f}")
            return True

        except Exception as e:
            print(f"❌ Критическая ошибка обучения: {e}")
            traceback.print_exc()
            return False

    def _update_training_metadata(self, new_samples_count):
        """Обновление метаданных обучения"""
        current_total = self.model_metadata.get('total_training_samples', 0)
        self.model_metadata['total_training_samples'] = current_total + new_samples_count
        self.model_metadata['training_sessions'] = self.model_metadata.get('training_sessions', 0) + 1
        self.model_metadata['last_training'] = datetime.now().isoformat()

        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'samples': new_samples_count,
            'total_samples': self.model_metadata['total_training_samples']
        })

    def _save_models(self):
        """Сохранение моделей"""
        try:
            models_to_save = {
                'main_model': self.model,
                'probabilistic_model': self.probabilistic_model,
                'advanced_model': self.advanced_model,
                'scaler': self.scaler,
                'imputer': self.imputer,
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'model_metadata': self.model_metadata
            }

            joblib.dump(models_to_save, self.model_path)
            print("💾 Модели успешно сохранены")
            return True

        except Exception as e:
            print(f"❌ Ошибка сохранения моделей: {e}")
            return False

    def predict_returns(self, prices_df, news_sentiment):
        """Предсказание доходности с учетом комиссий"""
        if not self.is_trained or self.advanced_model is None:
            print("⚠️ Модель не обучена, возвращаем нулевые предсказания")
            return {}

        features_dict = self.create_advanced_features(prices_df, news_sentiment)

        predictions = {}
        for ticker, features in features_dict.items():
            try:
                if features:
                    feature_vector = np.array(list(features.values())).reshape(1, -1)
                    feature_vector_imputed = self.imputer.transform(feature_vector)
                    feature_vector_scaled = self.scaler.transform(feature_vector_imputed)
                    predicted_return = self.advanced_model.predict(feature_vector_scaled)[0]

                    predictions[ticker] = {
                        'predicted_return': float(predicted_return),
                        'confidence': 0.7,
                        'sector': self.get_sector_for_ticker(ticker)
                    }

            except Exception as e:
                print(f"⚠️ Ошибка предсказания для {ticker}: {e}")
                continue

        return predictions

    def optimize_portfolio(self, predictions, prices_df, risk_aversion=1.0):
        """Оптимизация портфеля с учетом комиссий"""
        tickers = [t for t in predictions.keys() if t in prices_df.columns]
        if len(tickers) < 2:
            print("⚠️ Недостаточно тикеров для оптимизации")
            return self._get_equal_weights(tickers)

        try:
            returns_data = prices_df[tickers].pct_change().dropna()
            if len(returns_data) < 10:
                return self._get_equal_weights(tickers)

            cov_matrix = returns_data.cov().values * 252
            expected_returns = np.array([predictions[t]['predicted_return'] for t in tickers])

            def objective(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

                # ✅ ШТРАФ ЗА ЧАСТУЮ ПЕРЕБАЛАНСИРОВКУ (учет комиссий)
                turnover_penalty = np.sum(np.abs(weights)) * 0.001  # Штраф за оборот

                sharpe_ratio = (portfolio_return - turnover_penalty) / (portfolio_risk + 1e-8)
                return -sharpe_ratio

            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = [(0, 0.3) for _ in range(len(tickers))]
            initial_weights = np.array([1.0 / len(tickers)] * len(tickers))

            result = minimize(
                objective, initial_weights,
                method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': 100}
            )

            if result.success:
                weights = {ticker: weight for ticker, weight in zip(tickers, result.x) if weight > 0.01}
                total = sum(weights.values())
                if total > 0:
                    weights = {k: v / total for k, v in weights.items()}
                    print(f"✅ Оптимизация завершена с учетом комиссий, выбрано {len(weights)} активов")
                    return weights

        except Exception as e:
            print(f"⚠️ Ошибка оптимизации: {e}")

        return self._get_proportional_weights(predictions, tickers)

    def _get_equal_weights(self, tickers):
        """Равномерное распределение"""
        n = len(tickers)
        return {ticker: 1.0 / n for ticker in tickers} if n > 0 else {}

    def _get_proportional_weights(self, predictions, tickers):
        """Пропорциональное распределение по ожидаемой доходности"""
        positive_returns = {t: max(0.01, predictions[t]['predicted_return']) for t in tickers}
        total_positive = sum(positive_returns.values())
        if total_positive > 0:
            return {t: r / total_positive for t, r in positive_returns.items()}
        else:
            return self._get_equal_weights(tickers)

    def get_model_status(self):
        """Получение статуса модели"""
        return {
            'is_trained': self.is_trained,
            'training_sessions': len(self.training_history),
            'total_samples': self.model_metadata.get('total_training_samples', 0),
            'last_training': self.model_metadata.get('last_training', 'Never')
        }


# ==================== УЛУЧШЕННЫЙ КЛАСС ВИРТУАЛЬНОГО ПОРТФЕЛЯ С КОМИССИЯМИ ====================
class CommissionAwareVirtualPortfolio:
    def __init__(self, initial_cash=10000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.transaction_history = []
        self.commission_history = []
        self.daily_commissions = {}
        self.portfolio_file = 'portfolio.json'
        self.commission_manager = BrokerCommissionManager()
        self.load_portfolio()

    def load_portfolio(self):
        """Загрузка портфеля из файла"""
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, 'r') as f:
                    data = json.load(f)
                    self.cash = data.get('cash', self.initial_cash)
                    self.positions = data.get('positions', {})
                    self.transaction_history = data.get('transaction_history', [])
                    self.commission_history = data.get('commission_history', [])
                    self.daily_commissions = data.get('daily_commissions', {})
                print("✅ Портфель с комиссиями загружен")
            except Exception as e:
                print(f"Ошибка загрузки портфеля: {e}")

    def save_portfolio(self):
        """Сохранение портфеля в файл"""
        try:
            data = {
                'cash': self.cash,
                'positions': self.positions,
                'transaction_history': self.transaction_history[-100:],
                'commission_history': self.commission_history[-100:],
                'daily_commissions': self.daily_commissions,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.portfolio_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения портфеля: {e}")

    def _get_today_date(self):
        """Получение сегодняшней даты в формате строки"""
        return datetime.now().strftime('%Y-%m-%d')

    def _update_daily_commissions(self, commission_amount):
        """Обновление дневных комиссий"""
        today = self._get_today_date()
        if today not in self.daily_commissions:
            self.daily_commissions[today] = 0
        self.daily_commissions[today] += commission_amount

    def execute_trade(self, sector, ticker, action, price, quantity):
        """Исполнение сделки с учетом комиссий"""
        try:
            if quantity <= 0:
                return False

            trade_amount = price * quantity

            # Проверка минимальной суммы сделки
            min_trade_amount = self.commission_manager.commission_config['min_trade_amount']
            if trade_amount < min_trade_amount:
                print(
                    f"⚠️ Сделка {ticker} пропущена: сумма {trade_amount:.2f} ₽ меньше минимальной {min_trade_amount} ₽")
                return False

            # Расчет комиссий
            commission_info = self.commission_manager.calculate_trade_commission(trade_amount)
            total_cost = trade_amount + commission_info['total_commission'] if action == 'BUY' else trade_amount
            net_proceeds = trade_amount - commission_info['total_commission'] if action == 'SELL' else 0

            if action == 'BUY':
                if self.cash >= total_cost:
                    # Списываем стоимость акций и комиссии
                    self.cash -= total_cost

                    if ticker in self.positions:
                        total_quantity = self.positions[ticker]['quantity'] + quantity
                        total_cost_basis = (self.positions[ticker]['avg_price'] * self.positions[ticker][
                            'quantity']) + trade_amount
                        self.positions[ticker]['avg_price'] = total_cost_basis / total_quantity
                        self.positions[ticker]['quantity'] = total_quantity
                    else:
                        self.positions[ticker] = {
                            'quantity': quantity,
                            'avg_price': price,
                            'sector': sector
                        }

                    trade_record = {
                        'timestamp': datetime.now().isoformat(),
                        'sector': sector,
                        'ticker': ticker,
                        'action': action,
                        'quantity': quantity,
                        'price': price,
                        'trade_amount': trade_amount,
                        'commission': commission_info['total_commission'],
                        'total_cost': total_cost,
                        'type': 'BUY'
                    }

                    self.transaction_history.append(trade_record)
                    self.commission_history.append(commission_info)
                    self._update_daily_commissions(commission_info['total_commission'])
                    self.save_portfolio()

                    print(
                        f"✅ BUY: {ticker} {quantity} шт по ₽{price:.2f}, Комиссия: ₽{commission_info['total_commission']:.2f}")
                    return True

            elif action == 'SELL':
                if ticker in self.positions and self.positions[ticker]['quantity'] >= quantity:
                    # Зачисляем выручку за вычетом комиссий
                    self.cash += net_proceeds
                    self.positions[ticker]['quantity'] -= quantity

                    if self.positions[ticker]['quantity'] == 0:
                        del self.positions[ticker]

                    trade_record = {
                        'timestamp': datetime.now().isoformat(),
                        'sector': sector,
                        'ticker': ticker,
                        'action': action,
                        'quantity': quantity,
                        'price': price,
                        'trade_amount': trade_amount,
                        'commission': commission_info['total_commission'],
                        'net_proceeds': net_proceeds,
                        'type': 'SELL'
                    }

                    self.transaction_history.append(trade_record)
                    self.commission_history.append(commission_info)
                    self._update_daily_commissions(commission_info['total_commission'])
                    self.save_portfolio()

                    print(
                        f"✅ SELL: {ticker} {quantity} шт по ₽{price:.2f}, Комиссия: ₽{commission_info['total_commission']:.2f}")
                    return True

            return False

        except Exception as e:
            print(f"❌ Ошибка исполнения сделки: {e}")
            return False

    def get_portfolio_value(self, current_prices):
        """Расчет текущей стоимости портфеля"""
        total_value = self.cash

        for ticker, position in self.positions.items():
            if ticker in current_prices:
                total_value += current_prices[ticker] * position['quantity']

        return total_value

    def get_commission_statistics(self):
        """Получение статистики по комиссиям"""
        today = self._get_today_date()
        total_commissions = sum(self.daily_commissions.values())
        today_commission = self.daily_commissions.get(today, 0)

        # Расчет налога на реализованную прибыль
        realized_profit = 0
        for trade in self.transaction_history:
            if trade['type'] == 'SELL':
                # Здесь можно добавить расчет реальной прибыли
                pass

        tax = self.commission_manager.calculate_tax(realized_profit)

        return {
            'total_commissions': total_commissions,
            'today_commission': today_commission,
            'estimated_tax': tax,
            'commission_config': self.commission_manager.commission_config
        }

    def rebalance_portfolio(self, target_weights, current_prices, max_position_size=0.3):
        """Ребалансировка портфеля с учетом комиссий"""
        total_value = self.get_portfolio_value(current_prices)
        if total_value == 0 or not target_weights:
            return []

        executed_trades = []
        today_trades = []

        # Сначала продаем избыточные позиции
        for ticker in list(self.positions.keys()):
            if ticker in current_prices and current_prices[ticker] > 0:
                current_quantity = self.positions[ticker]['quantity']
                current_value = current_quantity * current_prices[ticker]
                current_weight = current_value / total_value
                target_weight = target_weights.get(ticker, 0)

                # Более консервативный порог для продажи с учетом комиссий
                if current_weight > target_weight * 1.1:  # Увеличили порог до 10%
                    target_value = target_weight * total_value
                    sell_value = current_value - target_value
                    sell_quantity = int(sell_value / current_prices[ticker])

                    if sell_quantity > 0 and sell_quantity <= current_quantity:
                        success = self.execute_trade(
                            self.positions[ticker].get('sector', 'Общие'),
                            ticker, 'SELL', current_prices[ticker], sell_quantity
                        )
                        if success:
                            executed_trades.append(f"SELL {ticker} {sell_quantity} шт")
                            today_trades.append({'amount': sell_quantity * current_prices[ticker]})

        # Затем покупаем недостающие позиции (более консервативно)
        for ticker, target_weight in target_weights.items():
            if target_weight > 0 and ticker in current_prices and current_prices[ticker] > 0:
                current_quantity = self.positions.get(ticker, {}).get('quantity', 0)
                current_value = current_quantity * current_prices[ticker]
                current_weight = current_value / total_value

                # Более консервативный порог для покупки с учетом комиссий
                if current_weight < target_weight * 0.9:  # Увеличили порог до 10%
                    target_value = target_weight * total_value
                    buy_value = target_value - current_value

                    # Резервируем дополнительные средства на комиссии (5%)
                    available_cash = self.cash * 0.95
                    actual_buy_value = min(buy_value, available_cash)
                    buy_quantity = int(actual_buy_value / current_prices[ticker])

                    if buy_quantity > 0:
                        sector = next((s for s, t in RUSSIAN_SECTORS.items() if ticker in t), 'Общие')
                        success = self.execute_trade(
                            sector, ticker, 'BUY', current_prices[ticker], buy_quantity
                        )
                        if success:
                            executed_trades.append(f"BUY {ticker} {buy_quantity} шт")
                            today_trades.append({'amount': buy_quantity * current_prices[ticker]})

        # Применяем минимальную дневную комиссию
        if today_trades:
            today_commission = self.daily_commissions.get(self._get_today_date(), 0)
            min_daily = self.commission_manager.commission_config['min_commission_per_day']

            if today_commission < min_daily:
                additional_commission = min_daily - today_commission
                self.cash -= additional_commission
                self._update_daily_commissions(additional_commission)
                print(f"⚠️ Применена минимальная дневная комиссия: ₽{additional_commission:.2f}")

        return executed_trades


# ==================== ВЕБ-ИНТЕРФЕЙС С ОТОБРАЖЕНИЕМ КОМИССИЙ ====================
app = Flask(__name__)


class CommissionAwareWebInterface:
    def __init__(self):
        self.current_recommendations = {}
        self.news_items = []
        self.portfolio_value = 0
        self.is_running = True
        self.portfolio_weights = {}
        self.model_info = {}
        self.commission_stats = {}

    def update_data(self, recommendations, news_df, portfolio_value, portfolio_weights=None,
                    model_info=None, commission_stats=None):
        """Обновление данных для отображения"""
        self.current_recommendations = recommendations
        self.news_items = news_df.to_dict('records') if not news_df.empty else []
        self.portfolio_value = portfolio_value
        self.portfolio_weights = portfolio_weights or {}
        self.model_info = model_info or {}
        self.commission_stats = commission_stats or {}

    def get_html_template(self):
        """HTML шаблон веб-интерфейса с отображением комиссий"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Анализ российских акций с ИИ (с учетом комиссий)</title>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1400px; margin: 0 auto; }
                .header { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .recommendations { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-bottom: 20px; }
                .sector-card { background: white; padding: 15px; border-radius: 8px; border-left: 5px solid #ddd; }
                .positive { border-left-color: #4CAF50 !important; background: #f1f8e9 !important; }
                .negative { border-left-color: #f44336 !important; background: #ffebee !important; }
                .neutral { border-left-color: #2196F3 !important; background: #e3f2fd !important; }
                .news-item { background: white; padding: 15px; margin-bottom: 10px; border-radius: 8px; border-left: 4px solid #2196F3; }
                .news-new { background: #e3f2fd !important; border-left-color: #FF9800 !important; }
                .controls { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .btn { padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
                .btn-stop { background: #f44336; }
                .portfolio { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .weights-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                .weights-table th, .weights-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                .weights-table th { background: #f5f5f5; }
                .model-info { background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 10px 0; }
                .commission-info { background: #fff3e0; padding: 15px; border-radius: 8px; margin: 10px 0; }
                .profit-positive { color: #4CAF50; font-weight: bold; }
                .profit-negative { color: #f44336; font-weight: bold; }
                .commission-warning { color: #ff9800; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 Анализ российских акций с ИИ (с учетом комиссий)</h1>
                    <p>Последнее обновление: {{ current_time }}</p>
                </div>

                <div class="controls">
                    <button class="btn" onclick="location.reload()">Обновить</button>
                    <button class="btn btn-stop" onclick="stopProgram()">Остановить программу</button>
                </div>

                <div class="portfolio">
                    <h2>💼 Виртуальный портфель</h2>
                    <p><strong>Текущая стоимость:</strong> {{ "₽{:,.2f}".format(portfolio_value) }}</p>
                    {% set profit = portfolio_value - 10000 %}
                    {% set profit_percent = (portfolio_value / 10000 - 1) * 100 %}
                    <p><strong>Доходность:</strong> 
                        <span class="{{ 'profit-positive' if profit >= 0 else 'profit-negative' }}">
                            {{ "₽{:,.2f}".format(profit) }} ({{ "{:.2f}%".format(profit_percent) }})
                        </span>
                    </p>

                    {% if commission_stats %}
                    <div class="commission-info">
                        <h3>💸 Комиссии и налоги</h3>
                        <p><strong>Всего комиссий:</strong> {{ "₽{:,.2f}".format(commission_stats.total_commissions) }}</p>
                        <p><strong>Комиссии за сегодня:</strong> {{ "₽{:,.2f}".format(commission_stats.today_commission) }}</p>
                        <p><strong>Расчетный налог:</strong> {{ "₽{:,.2f}".format(commission_stats.estimated_tax) }}</p>
                        <p><strong>Комиссия брокера:</strong> {{ "%.3f"|format(commission_stats.commission_config.commission_percent) }}%</p>
                        <p><strong>Мин. комиссия за сделку:</strong> {{ "₽{:,.2f}".format(commission_stats.commission_config.min_commission_per_trade) }}</p>
                        <p><strong>Мин. комиссия за день:</strong> {{ "₽{:,.2f}".format(commission_stats.commission_config.min_commission_per_day) }}</p>
                    </div>
                    {% endif %}

                    {% if portfolio_weights %}
                    <h3>📊 Распределение портфеля:</h3>
                    <table class="weights-table">
                        <tr><th>Тикер</th><th>Вес</th><th>Действие</th></tr>
                        {% for ticker, weight in portfolio_weights.items() %}
                        <tr>
                            <td>{{ ticker }}</td>
                            <td>{{ "{:.1f}%".format(weight * 100) }}</td>
                            <td>
                                {% if weight > 0.1 %}🟢 Сильная покупка
                                {% elif weight > 0.05 %}🟢 Покупка  
                                {% elif weight > 0.02 %}🟡 Удержание
                                {% else %}🔴 Продажа{% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </table>
                    {% endif %}
                </div>

                {% if model_info %}
                <div class="model-info">
                    <h3>🤖 Информация о модели</h3>
                    <p>Обучений: {{ model_info.training_sessions }}, Примеров: {{ model_info.total_samples }}</p>
                    <p>Последнее обучение: {{ model_info.last_training }}</p>
                    <p class="commission-warning">⚠️ Модель обучена с учетом комиссий брокера</p>
                </div>
                {% endif %}

                <h2>📈 Рекомендации по секторам</h2>
                <div class="recommendations">
                    {% for sector, rec in recommendations.items() %}
                    <div class="sector-card {{ 'positive' if rec > 1 else 'negative' if rec < -1 else 'neutral' }}">
                        <h3>{{ sector }}</h3>
                        <p><strong>Рекомендация:</strong> {{ rec }}/5</p>
                        <p><em>{{ get_recommendation_text(rec) }}</em></p>
                    </div>
                    {% endfor %}
                </div>

                <h2>📰 Последние новости</h2>
                <div id="news">
                    {% for news in news_items %}
                    <div class="news-item {{ 'news-new' if is_new_news(news.timestamp) else '' }}">
                        <h4>{{ news.title }}</h4>
                        <p><strong>Источник:</strong> {{ news.source }} | <strong>Дата:</strong> {{ news.published }}</p>
                        <p>{{ news.summary[:200] }}{% if news.summary|length > 200 %}...{% endif %}</p>
                    </div>
                    {% endfor %}
                </div>
            </div>

            <script>
                function stopProgram() {
                    fetch('/stop', { method: 'POST' })
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                alert('Программа останавливается...');
                            }
                        });
                }

                setTimeout(() => location.reload(), 300000);
            </script>
        </body>
        </html>
        '''

    def run_server(self, host='0.0.0.0', port=5000):
        """Запуск веб-сервера"""

        @app.route('/')
        def index():
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            def get_recommendation_text(rec):
                if rec >= 3:
                    return 'Сильная покупка'
                elif rec >= 1:
                    return 'Умеренная покупка'
                elif rec == 0:
                    return 'Нейтрально'
                elif rec >= -2:
                    return 'Умеренная продажа'
                else:
                    return 'Сильная продажа'

            def is_new_news(timestamp):
                if isinstance(timestamp, str):
                    try:
                        news_time = datetime.fromisoformat(timestamp.replace('Z', ''))
                    except:
                        return False
                else:
                    news_time = timestamp
                return (datetime.now() - news_time).total_seconds() < 3600

            template = self.get_html_template()
            return render_template_string(
                template,
                recommendations=self.current_recommendations,
                news_items=self.news_items[-15:],
                current_time=current_time,
                portfolio_value=self.portfolio_value,
                portfolio_weights=self.portfolio_weights,
                model_info=self.model_info,
                commission_stats=self.commission_stats,
                get_recommendation_text=get_recommendation_text,
                is_new_news=is_new_news
            )

        @app.route('/stop', methods=['POST'])
        def stop_program():
            self.is_running = False
            return {'success': True}

        print(f"🚀 Веб-интерфейс с учетом комиссий доступен по адресу: http://{host}:{port}")
        app.run(host=host, port=port, debug=False, use_reloader=False)


# ==================== ОСНОВНОЙ КЛАСС СИСТЕМЫ С УЧЕТОМ КОМИССИЙ ====================
class CommissionAwareStockAnalysisSystem:
    def __init__(self):
        self.news_collector = NewsCollector()
        self.stock_collector = CorrectedStockDataCollector()
        self.news_analyzer = NewsAnalyzer()
        self.enhanced_ai_model = CommissionAwareStockAIModel()
        self.portfolio = CommissionAwareVirtualPortfolio()
        self.web_interface = CommissionAwareWebInterface()

        self.news_backup_file = 'news_backup.csv'
        self.prices_backup_file = 'prices_backup.csv'
        self.recommendations_file = 'RECOM.csv'

    def run_enhanced_analysis_cycle(self):
        """Цикл анализа с учетом комиссий"""
        print("\n" + "=" * 60)
        print(f"🔍 Запуск анализа с учетом комиссий: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        try:
            model_status = self.enhanced_ai_model.get_model_status()
            print(f"🤖 Статус модели: {model_status}")

            print("📰 Шаг 1: Сбор новостей...")
            news_df = self.news_collector.fetch_all_news()

            print("📊 Шаг 2: Сбор данных об акциях...")
            all_tickers = [ticker for sector_tickers in RUSSIAN_SECTORS.values() for ticker in sector_tickers]

            historical_prices_df = self.stock_collector.get_historical_data(all_tickers, days=90)
            current_prices = self.stock_collector.get_current_prices(all_tickers)

            if historical_prices_df.empty:
                print("❌ Нет исторических данных для обучения, пропускаем цикл")
                return False

            if not current_prices:
                print("⚠️ Не удалось получить текущие цены, используем последние исторические")
                current_prices = historical_prices_df.iloc[-1].to_dict()

            print("💰 ТЕКУЩИЕ ЦЕНЫ ДЛЯ ПОРТФЕЛЯ:")
            for ticker, price in list(current_prices.items())[:5]:
                print(f"  {ticker}: {price:.2f} ₽")

            print("🔍 Шаг 3: Анализ новостей...")
            news_recommendations = self.news_analyzer.predict_sector_sentiment(news_df)

            print("🎯 Шаг 4: Обучение AI моделей с учетом комиссий...")
            news_sentiment_by_date = {datetime.now().date(): news_recommendations}

            incremental_learning = self.enhanced_ai_model.is_trained
            training_successful = self.enhanced_ai_model.train_models(
                historical_prices_df, news_sentiment_by_date, incremental=incremental_learning
            )

            if training_successful:
                training_type = "дообучение" if incremental_learning else "первоначальное обучение"
                print(f"✅ Модели успешно прошли {training_type} с учетом комиссий")
            else:
                print("⚠️ Обучение не удалось, используем базовые рекомендации")

            print("🧠 Шаг 5: AI анализ и оптимизация портфеля с учетом комиссий...")

            if self.enhanced_ai_model.is_trained:
                ai_predictions = self.enhanced_ai_model.predict_returns(historical_prices_df, news_recommendations)
                optimal_weights = self.enhanced_ai_model.optimize_portfolio(ai_predictions, historical_prices_df)

                if optimal_weights:
                    rebalancing_trades = self.portfolio.rebalance_portfolio(optimal_weights, current_prices)
                    if rebalancing_trades:
                        print(f"⚖️ Выполнено сделок: {len(rebalancing_trades)}")
                        for trade in rebalancing_trades:
                            print(f"  {trade}")
                else:
                    optimal_weights = {}
                    print("⚠️ Не удалось оптимизировать портфель")
            else:
                optimal_weights = {}
                print("⚠️ Модель не обучена, пропускаем оптимизацию")

            enhanced_recommendations = self._enhance_recommendations_with_optimization(
                news_recommendations, optimal_weights
            )

            portfolio_value = self.portfolio.get_portfolio_value(current_prices)
            commission_stats = self.portfolio.get_commission_statistics()
            self.save_backup_data(news_df, historical_prices_df, enhanced_recommendations)

            model_info = {
                'training_sessions': self.enhanced_ai_model.model_metadata.get('training_sessions', 0),
                'total_samples': self.enhanced_ai_model.model_metadata.get('total_training_samples', 0),
                'last_training': self.enhanced_ai_model.model_metadata.get('last_training', 'Never')
            }

            self.web_interface.update_data(
                enhanced_recommendations, news_df, portfolio_value, optimal_weights,
                model_info, commission_stats
            )

            self._print_enhanced_results(enhanced_recommendations, optimal_weights, portfolio_value, commission_stats)

            return True

        except Exception as e:
            print(f"💥 Ошибка в цикле анализа: {e}")
            traceback.print_exc()
            return False

    def _enhance_recommendations_with_optimization(self, news_recommendations, optimal_weights):
        """Улучшение рекомендаций на основе оптимизации портфеля"""
        enhanced_recommendations = news_recommendations.copy()

        if optimal_weights:
            sector_weights = {}
            for ticker, weight in optimal_weights.items():
                sector = self.enhanced_ai_model.get_sector_for_ticker(ticker)
                if sector not in sector_weights:
                    sector_weights[sector] = 0
                sector_weights[sector] += weight

            if sector_weights:
                max_weight = max(sector_weights.values())
                for sector in enhanced_recommendations:
                    weight = sector_weights.get(sector, 0)
                    if max_weight > 0:
                        optimization_score = (weight / max_weight) * 10 - 5
                        enhanced_recommendations[sector] = int(
                            0.4 * enhanced_recommendations[sector] + 0.6 * optimization_score
                        )
                        enhanced_recommendations[sector] = max(-5, min(5, enhanced_recommendations[sector]))

        return enhanced_recommendations

    def _print_enhanced_results(self, recommendations, portfolio_weights, portfolio_value, commission_stats):
        """Вывод результатов анализа с комиссиями"""
        print("\n📈 РЕЗУЛЬТАТЫ АНАЛИЗА С УЧЕТОМ КОМИССИЙ:")
        print("-" * 60)

        print("🎯 РЕКОМЕНДАЦИИ ПО СЕКТОРАМ:")
        for sector, rec in sorted(recommendations.items(), key=lambda x: x[1], reverse=True):
            emoji = "🟢" if rec > 1 else "🔴" if rec < -1 else "🟡"
            print(f"{emoji} {sector:30} {rec:+2d}/5")

        if portfolio_weights:
            print(f"\n📊 ОПТИМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ:")
            for ticker, weight in sorted(portfolio_weights.items(), key=lambda x: x[1], reverse=True)[:8]:
                print(f"  {ticker:10} {weight * 100:5.1f}%")

        initial_value = 10000
        profit = portfolio_value - initial_value
        profit_percent = (portfolio_value / initial_value - 1) * 100

        print(f"\n💼 ПОРТФЕЛЬ: ₽{portfolio_value:,.0f}")
        profit_color = "🟢" if profit >= 0 else "🔴"
        print(f"📈 ДОХОДНОСТЬ: {profit_color} ₽{profit:+,.0f} ({profit_percent:+.2f}%)")

        print(f"\n💸 КОМИССИИ:")
        print(f"  Всего комиссий: ₽{commission_stats['total_commissions']:,.2f}")
        print(f"  Комиссии за сегодня: ₽{commission_stats['today_commission']:,.2f}")
        print(f"  Расчетный налог: ₽{commission_stats['estimated_tax']:,.2f}")

        model_info = self.enhanced_ai_model.model_metadata
        print(f"🤖 МОДЕЛЬ: {model_info.get('training_sessions', 0)} обучений, "
              f"{model_info.get('total_training_samples', 0)} примеров")

        print("✅ АНАЛИЗ С УЧЕТОМ КОМИССИЙ ЗАВЕРШЕН!")

    def save_backup_data(self, news_df, prices_df, recommendations):
        """Сохранение резервных копий данных"""
        try:
            if not news_df.empty:
                news_df.to_csv(self.news_backup_file, index=False, encoding='utf-8-sig')

            if not prices_df.empty:
                prices_df.to_csv(self.prices_backup_file, index=False, encoding='utf-8-sig')

            rec_df = pd.DataFrame(list(recommendations.items()), columns=['Сектор', 'Рекомендация'])
            rec_df['Дата'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rec_df.to_csv(self.recommendations_file, index=False, encoding='utf-8-sig')

            print("💾 Данные сохранены")

        except Exception as e:
            print(f"Ошибка сохранения данных: {e}")

    def run_continuous_analysis(self, interval_minutes=3):
        """Запуск непрерывного анализа"""
        print("🚀 Запуск системы анализа с учетом комиссий брокера...")
        print(f"💰 Начальный капитал: ₽{self.portfolio.initial_cash:,.0f}")

        web_thread = threading.Thread(target=self.web_interface.run_server)
        web_thread.daemon = True
        web_thread.start()

        cycle_count = 0
        while self.web_interface.is_running:
            try:
                success = self.run_enhanced_analysis_cycle()
                cycle_count += 1

                if success:
                    print(f"\n♻️ Цикл {cycle_count} завершен. Ожидание {interval_minutes} минут...")
                    wait_seconds = interval_minutes * 60
                    for i in range(wait_seconds):
                        if not self.web_interface.is_running:
                            break
                        time.sleep(1)
                else:
                    print("💤 Ожидание 10 минут перед повторной попыткой...")
                    time.sleep(600)

            except KeyboardInterrupt:
                print("\n⚠️ Программа прервана пользователем")
                break
            except Exception as e:
                print(f"💥 Ошибка: {e}")
                time.sleep(600)

        print("🛑 Программа остановлена")


# ==================== ЗАПУСК ПРОГРАММЫ ====================
if __name__ == "__main__":
    system = CommissionAwareStockAnalysisSystem()
    system.run_continuous_analysis(interval_minutes=3)