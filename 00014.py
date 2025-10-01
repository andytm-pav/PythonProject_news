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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import re
import threading
from flask import Flask, render_template_string, request
import json
import traceback

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

RUSSIAN_FEEDS = {
    "Ведомости": "https://www.vedomosti.ru/rss/news",
    "Коммерсант": "https://www.kommersant.ru/RSS/news.xml",
    "Интерфакс": "https://www.interfax.ru/rss.asp",
    "ТАСС": "https://tass.ru/rss/v2.xml",
    "Лента.ру": "https://lenta.ru/rss/news",
    "Финам": "https://www.finam.ru/analysis/conews/rsspoint/",
    "БФМ": "https://www.bfm.ru/news.rss?rubric=19",
    "Альта_закон": "http://www.alta.ru/rss/laws_news/",
    "Финам мировых": "https://www.finam.ru/international/advanced/rsspoint/",
    "Инвестирование ком": "https://ru.investing.com/rss/news_356.rss",
    "Инвестирование ком сырьё": "https://ru.investing.com/rss/news_11.rss",
    "Инвестирование ком Экономики": "https://ru.investing.com/rss/news_14.rss",
    "Инвестирование ком фондовый": "https://ru.investing.com/rss/news_25.rss",
}

# ==================== КЛАСС ДЛЯ СБОРА НОВОСТЕЙ ====================
class NewsCollector:
    def __init__(self):
        self.feeds = RUSSIAN_FEEDS

    def fetch_all_news(self, max_items_per_feed=20):
        """Сбор новостей из всех RSS-лент с обработкой ошибок"""
        all_news = []

        for source, url in self.feeds.items():
            try:
                print(f"Загрузка новостей из {source}...")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }

                feed = feedparser.parse(url)
                time.sleep(1)

                if not feed.entries:
                    print(f"  Нет новостей в {source}")
                    continue

                items_added = 0
                for entry in feed.entries[:max_items_per_feed]:
                    try:
                        title = getattr(entry, 'title', '').strip()
                        if not title:
                            continue

                        published = getattr(entry, 'published', '')
                        if not published:
                            published = getattr(entry, 'updated', datetime.now().isoformat())

                        news_item = {
                            'source': source,
                            'title': title,
                            'link': getattr(entry, 'link', ''),
                            'published': published,
                            'published_parsed': getattr(entry, 'published_parsed', None),
                            'summary': getattr(entry, 'summary', '')[:500],
                            'timestamp': datetime.now()
                        }

                        all_news.append(news_item)
                        items_added += 1

                    except Exception as e:
                        continue

                print(f"  Добавлено {items_added} новостей из {source}")

            except Exception as e:
                print(f"Ошибка при загрузке из {source}: {e}")
                continue

        news_df = pd.DataFrame(all_news)
        if not news_df.empty:
            news_df = news_df.drop_duplicates(subset=['title'])
            news_df = news_df.reset_index(drop=True)

        print(f"Всего собрано новостей: {len(news_df)}")
        return news_df


# ==================== КЛАСС ДЛЯ ПОЛУЧЕНИЯ ДАННЫХ О АКЦИЯХ ====================
class StockDataCollector:
    def __init__(self):
        self.base_url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities"

    def get_stock_prices(self, tickers, days=30):
        """Получение данных об акциях с Московской биржи"""
        stock_data = {}
        successful_downloads = 0

        for ticker in tickers:
            try:
                url = f"{self.base_url}/{ticker}/candles.json"
                from_date = (datetime.now() - timedelta(days=days * 2)).strftime('%Y-%m-%d')

                params = {
                    'from': from_date,
                    'till': datetime.now().strftime('%Y-%m-%d'),
                    'interval': 24
                }

                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    candles = data.get('candles', {}).get('data', [])

                    if candles:
                        dates = []
                        prices = []

                        for candle in candles:
                            date_str = candle[6]
                            close_price = candle[5]

                            if date_str and close_price:
                                dates.append(date_str)
                                prices.append(close_price)

                        if dates and prices:
                            series = pd.Series(prices, index=dates)
                            stock_data[ticker] = series
                            successful_downloads += 1

                time.sleep(0.5)

            except Exception as e:
                print(f"Ошибка загрузки {ticker}: {e}")
                continue

        print(f"Успешно загружено данных для {successful_downloads} тикеров")

        if stock_data:
            prices_df = pd.DataFrame(stock_data)
            prices_df = prices_df.ffill().bfill()
            return prices_df
        else:
            return self.create_test_data(tickers, days)

    def create_test_data(self, tickers, days):
        """Создание тестовых данных если не удалось получить реальные"""
        print("Создание тестовых данных...")
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
            returns = np.random.normal(0.001, 0.02, days)

            prices = [base_price]
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 0.01))

            prices_df[ticker] = prices

        return prices_df


# ==================== КЛАСС АНАЛИЗА НОВОСТЕЙ ====================
class NewsAnalyzer:
    def __init__(self):
        self.sector_keywords = self._build_sector_keywords()
        self.sentiment_words = self._build_sentiment_lexicon()

    def _build_sector_keywords(self):
        return {
            'Энергетика': ['нефть', 'газ', 'нефтегаз', 'энергетик', 'нефтяной', 'газовый', 'нефтедобыча',
                           'месторождение'],
            'Индустриальный': ['промышленност', 'завод', 'производств', 'индустриальн', 'оборудован'],
            'Базовые ресурсы и материалы': ['металл', 'сталь', 'никель', 'алюмин', 'медь', 'руд', 'горнодобывающ'],
            'Розничная и оптовая торговля': ['ритейл', 'магазин', 'торговля', 'розничн', 'покуп', 'продаж', 'товар'],
            'Медицина, фармацевтика, охрана здоровья': ['фарма', 'медицин', 'лекарств', 'препарат', 'витамин',
                                                        'здоровье', 'больни'],
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
# ==================== КЛАСС ВИРТУАЛЬНОГО ПОРТФЕЛЯ ====================
class VirtualPortfolio:
    def __init__(self, initial_cash=1000000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.transaction_history = []
        self.portfolio_file = 'portfolio.json'
        self.load_portfolio()

    def load_portfolio(self):
        """Загрузка портфеля из файла"""
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, 'r') as f:
                    data = json.load(f)
                    self.cash = data.get('cash', self.initial_cash)
                    self.positions = data.get('positions', {})
                print("✅ Портфель загружен")
            except Exception as e:
                print(f"Ошибка загрузки портфеля: {e}")

    def save_portfolio(self):
        """Сохранение портфеля в файл"""
        try:
            data = {
                'cash': self.cash,
                'positions': self.positions,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.portfolio_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения портфеля: {e}")

    def execute_trade(self, sector, ticker, action, price, quantity):
        """Исполнение сделки"""
        try:
            cost = price * quantity

            if action == 'BUY':
                if self.cash >= cost:
                    self.cash -= cost
                    if ticker in self.positions:
                        self.positions[ticker]['quantity'] += quantity
                        self.positions[ticker]['avg_price'] = (
                                                                      self.positions[ticker]['avg_price'] + price) / 2
                    else:
                        self.positions[ticker] = {
                            'quantity': quantity,
                            'avg_price': price,
                            'sector': sector
                        }

                    self.transaction_history.append({
                        'timestamp': datetime.now(),
                        'sector': sector,
                        'ticker': ticker,
                        'action': action,
                        'quantity': quantity,
                        'price': price,
                        'total': cost
                    })
                    self.save_portfolio()
                    return True

            elif action == 'SELL':
                if ticker in self.positions and self.positions[ticker]['quantity'] >= quantity:
                    self.cash += cost
                    self.positions[ticker]['quantity'] -= quantity

                    if self.positions[ticker]['quantity'] == 0:
                        del self.positions[ticker]

                    self.transaction_history.append({
                        'timestamp': datetime.now(),
                        'sector': sector,
                        'ticker': ticker,
                        'action': action,
                        'quantity': quantity,
                        'price': price,
                        'total': cost
                    })
                    self.save_portfolio()
                    return True

            return False

        except Exception as e:
            print(f"Ошибка исполнения сделки: {e}")
            return False

    def get_portfolio_value(self, current_prices):
        """Расчет текущей стоимости портфеля"""
        total_value = self.cash

        for ticker, position in self.positions.items():
            if ticker in current_prices:
                total_value += current_prices[ticker] * position['quantity']

        return total_value

# ==================== КЛАСС АВТОМАТИЧЕСКОЙ РЕБАЛАНСИРОВКИ ====================
class PortfolioRebalancer:
    def __init__(self, portfolio, threshold=0.05, strategy='calendar_absolute'):
        self.portfolio = portfolio
        self.threshold = threshold
        self.strategy = strategy
        self.last_rebalance_date = None
        self.target_weights = self._initialize_target_weights()

    def _initialize_target_weights(self):
        """Инициализация целевых весов на основе равномерного распределения по секторам"""
        sector_weights = {}
        sectors_count = len(RUSSIAN_SECTORS)

        for sector in RUSSIAN_SECTORS.keys():
            sector_weights[sector] = 1.0 / sectors_count

        return sector_weights

    def calculate_current_weights(self, current_prices):
        """Расчет текущих весов портфеля"""
        total_value = self.portfolio.get_portfolio_value(current_prices)
        if total_value == 0:
            return {}

        current_weights = {}
        sector_values = {sector: 0 for sector in RUSSIAN_SECTORS.keys()}

        for ticker, position in self.portfolio.positions.items():
            if ticker in current_prices:
                position_value = current_prices[ticker] * position['quantity']
                sector = position.get('sector', 'Общие')
                if sector in sector_values:
                    sector_values[sector] += position_value

        for sector, value in sector_values.items():
            current_weights[sector] = value / total_value

        return current_weights

    def needs_rebalancing(self, current_weights, current_date):
        """Определение необходимости ребалансировки"""

        if self.strategy.startswith('calendar'):
            if self.last_rebalance_date is None:
                return True

            days_since_rebalance = (current_date - self.last_rebalance_date).days

            if 'month' in self.strategy and days_since_rebalance >= 30:
                return True
            elif 'quarter' in self.strategy and days_since_rebalance >= 90:
                return True
            elif 'year' in self.strategy and days_since_rebalance >= 365:
                return True

            if 'absolute' in self.strategy or 'relative' in self.strategy:
                return self._check_threshold_deviation(current_weights)

        elif self.strategy == 'threshold':
            return self._check_threshold_deviation(current_weights)

        return False

    def _check_threshold_deviation(self, current_weights):
        """Проверка отклонения от целевых весов"""
        for sector, target_weight in self.target_weights.items():
            current_weight = current_weights.get(sector, 0)

            if self.strategy.endswith('absolute'):
                deviation = abs(current_weight - target_weight)
                if deviation > self.threshold:
                    return True

            elif self.strategy.endswith('relative'):
                if target_weight > 0:
                    relative_deviation = abs(current_weight - target_weight) / target_weight
                    if relative_deviation > self.threshold:
                        return True

        return False

    def generate_rebalancing_trades(self, current_weights, current_prices, ai_recommendations):
        """Генерация сделок для ребалансировки портфеля"""
        trades = []
        total_value = self.portfolio.get_portfolio_value(current_prices)

        if total_value == 0:
            return trades

        adjusted_target_weights = self._adjust_weights_with_ai(current_weights, ai_recommendations)

        for sector, target_weight in adjusted_target_weights.items():
            current_weight = current_weights.get(sector, 0)
            deviation = current_weight - target_weight

            if abs(deviation) > self.threshold:
                sector_tickers = RUSSIAN_SECTORS.get(sector, [])

                if deviation > 0 and sector_tickers:
                    sell_value = total_value * deviation * 0.8
                    trades.extend(self._generate_sell_trades(sector, sector_tickers, sell_value, current_prices))

                elif deviation < 0 and sector_tickers:
                    buy_value = total_value * abs(deviation) * 0.8
                    trades.extend(self._generate_buy_trades(sector, sector_tickers, buy_value, current_prices,
                                                            ai_recommendations.get(sector, 0)))

        return trades

    def _adjust_weights_with_ai(self, current_weights, ai_recommendations):
        """Корректировка целевых весов на основе AI рекомендаций"""
        adjusted_weights = self.target_weights.copy()

        for sector, recommendation in ai_recommendations.items():
            if sector in adjusted_weights:
                adjustment_factor = 1.0 + (recommendation * 0.06)
                adjusted_weights[sector] *= adjustment_factor

        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for sector in adjusted_weights:
                adjusted_weights[sector] /= total_weight

        return adjusted_weights

    def _generate_buy_trades(self, sector, tickers, total_buy_value, current_prices, sector_recommendation):
        """Генерация сделок на покупку"""
        trades = []

        if total_buy_value <= 0 or not tickers:
            return trades

        buy_per_ticker = total_buy_value / len(tickers)

        for ticker in tickers:
            if ticker in current_prices and current_prices[ticker] > 0:
                quantity = int(buy_per_ticker / current_prices[ticker])
                if quantity > 0:
                    trades.append({
                        'sector': sector,
                        'ticker': ticker,
                        'action': 'BUY',
                        'quantity': quantity,
                        'price': current_prices[ticker],
                        'reason': f'Ребалансировка сектора {sector} (AI реком: {sector_recommendation})'
                    })

        return trades

    def _generate_sell_trades(self, sector, tickers, total_sell_value, current_prices):
        """Генерация сделок на продажу"""
        trades = []

        if total_sell_value <= 0 or not tickers:
            return trades

        for ticker in tickers:
            if ticker in self.portfolio.positions and ticker in current_prices:
                position = self.portfolio.positions[ticker]
                position_value = position['quantity'] * current_prices[ticker]
                portfolio_value = self.portfolio.get_portfolio_value(current_prices)

                if portfolio_value > 0:
                    position_weight = position_value / portfolio_value
                    sell_quantity = int(position['quantity'] * (total_sell_value / portfolio_value) / position_weight)

                    if sell_quantity > 0:
                        trades.append({
                            'sector': sector,
                            'ticker': ticker,
                            'action': 'SELL',
                            'quantity': min(sell_quantity, position['quantity']),
                            'price': current_prices[ticker],
                            'reason': f'Ребалансировка сектора {sector}'
                        })

        return trades

    def execute_rebalancing(self, current_prices, ai_recommendations):
        """Выполнение ребалансировки портфеля"""
        current_weights = self.calculate_current_weights(current_prices)
        current_date = datetime.now()

        if self.needs_rebalancing(current_weights, current_date):
            print("🔄 Выполнение ребалансировки портфеля...")

            trades = self.generate_rebalancing_trades(current_weights, current_prices, ai_recommendations)

            executed_trades = []
            for trade in trades:
                success = self.portfolio.execute_trade(
                    trade['sector'],
                    trade['ticker'],
                    trade['action'],
                    trade['price'],
                    trade['quantity']
                )

                if success:
                    executed_trades.append(trade)
                    print(f"  {trade['action']} {trade['ticker']} {trade['quantity']} шт.")

            self.last_rebalance_date = current_date
            print(f"✅ Ребалансировка завершена. Исполнено сделок: {len(executed_trades)}")

            return executed_trades

        return []


# ==================== ВЕБ-ИНТЕРФЕЙС ====================
app = Flask(__name__)


class WebInterface:
    def __init__(self):
        self.current_recommendations = {}
        self.news_items = []
        self.portfolio_value = 0
        self.is_running = True

    def update_data(self, recommendations, news_df, portfolio_value):
        """Обновление данных для отображения"""
        self.current_recommendations = recommendations
        self.news_items = news_df.to_dict('records') if not news_df.empty else []
        self.portfolio_value = portfolio_value

    def get_html_template(self):
        """HTML шаблон веб-интерфейса"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Анализ российских акций с ИИ</title>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .recommendations { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-bottom: 20px; }
                .sector-card { background: white; padding: 15px; border-radius: 8px; border-left: 5px solid #ddd; }
                .positive { border-left-color: #4CAF50 !important; background: #f1f8e9 !important; }
                .negative { border-left-color: #f44336 !important; background: #ffebee !important; }
                .news-item { background: white; padding: 15px; margin-bottom: 10px; border-radius: 8px; border-left: 4px solid #2196F3; }
                .news-new { background: #e3f2fd !important; border-left-color: #FF9800 !important; }
                .controls { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .btn { padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 5px; cursor: pointer; }
                .btn-stop { background: #f44336; }
                .portfolio { background: white; padding: 20px; border-radius: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 Анализ российских акций с ИИ</h1>
                    <p>Последнее обновление: {{ current_time }}</p>
                </div>

                <div class="controls">
                    <button class="btn" onclick="location.reload()">Обновить</button>
                    <button class="btn btn-stop" onclick="stopProgram()">Остановить программу</button>
                </div>

                <div class="portfolio">
                    <h2>💼 Виртуальный портфель</h2>
                    <p><strong>Текущая стоимость:</strong> {{ "₽{:,.2f}".format(portfolio_value) }}</p>
                </div>

                <h2>📈 Рекомендации по секторам</h2>
                <div class="recommendations">
                    {% for sector, rec in recommendations.items() %}
                    <div class="sector-card {{ 'positive' if rec > 0 else 'negative' if rec < 0 else '' }}">
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

                // Автообновление каждые 5 минут
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
                if rec == 5:
                    return 'Максимальная покупка'
                elif rec >= 3:
                    return 'Сильная покупка'
                elif rec >= 1:
                    return 'Умеренная покупка'
                elif rec == 0:
                    return 'Нейтрально'
                elif rec >= -2:
                    return 'Умеренная продажа'
                elif rec >= -4:
                    return 'Сильная продажа'
                else:
                    return 'Максимальная продажа'

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
                news_items=self.news_items[-20:],
                current_time=current_time,
                portfolio_value=self.portfolio_value,
                get_recommendation_text=get_recommendation_text,
                is_new_news=is_new_news
            )

        @app.route('/stop', methods=['POST'])
        def stop_program():
            self.is_running = False
            return {'success': True}

        print(f"🚀 Веб-интерфейс доступен по адресу: http://{host}:{port}")
        app.run(host=host, port=port, debug=False, use_reloader=False)


# ==================== УЛУЧШЕННЫЙ КЛАСС ИИ МОДЕЛИ С ДООБУЧЕНИЕМ ====================
class EnhancedStockAIModel:
    def __init__(self, model_path='enhanced_stock_ai_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.probabilistic_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = []
        self.model_metadata = {}

        # Гарантированная инициализация всех компонентов
        self._initialize_all_components()

    def _initialize_all_components(self):
        """Гарантированная инициализация всех компонентов класса"""
        print("🔄 Инициализация ИИ модели...")

        # Сначала пытаемся загрузить существующую модель
        model_loaded = self._load_model_safe()

        if not model_loaded:
            print("📝 Создание новых моделей...")
            self._create_new_models()
        else:
            print("✅ Модель успешно загружена и готова к дообучению")

    def _load_model_safe(self):
        """Безопасная загрузка модели с обработкой различных форматов"""
        try:
            if not os.path.exists(self.model_path):
                print("📝 Файл модели не найден")
                return False

            print(f"🔄 Загрузка модели из {self.model_path}")
            model_data = joblib.load(self.model_path)

            # Проверяем разные форматы файлов моделей
            if isinstance(model_data, dict):
                # Новый формат с метаданными
                return self._load_new_format(model_data)
            else:
                # Старый формат (только модели)
                print("⚠️ Обнаружен старый формат модели, конвертируем...")
                return self._load_old_format(model_data)

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False

    def _load_new_format(self, model_data):
        """Загрузка нового формата модели с метаданными"""
        try:
            # Критически важные компоненты
            if 'main_model' not in model_data or 'probabilistic_model' not in model_data:
                print("❌ Отсутствуют основные модели в файле")
                return False

            self.model = model_data['main_model']
            self.probabilistic_model = model_data['probabilistic_model']

            # Опциональные компоненты
            self.scaler = model_data.get('scaler', StandardScaler())
            self.is_trained = model_data.get('is_trained', True)
            self.training_history = model_data.get('training_history', [])
            self.model_metadata = model_data.get('model_metadata', {})

            # Проверяем валидность загруженных моделей
            if self._validate_models():
                print("✅ Модели загружены и валидированы")
                return True
            else:
                print("❌ Загруженные модели невалидны")
                return False

        except Exception as e:
            print(f"❌ Ошибка загрузки нового формата: {e}")
            return False

    def _load_old_format(self, model_data):
        """Загрузка старого формата модели"""
        try:
            # Предполагаем, что model_data - это кортеж или список с моделями
            if isinstance(model_data, (list, tuple)) and len(model_data) >= 2:
                self.model = model_data[0]
                self.probabilistic_model = model_data[1]
                self.is_trained = True
                print("✅ Старый формат модели загружен")
                return True
            else:
                print("❌ Неизвестный формат модели")
                return False
        except Exception as e:
            print(f"❌ Ошибка загрузки старого формата: {e}")
            return False

    def _validate_models(self):
        """Проверка валидности загруженных моделей"""
        try:
            if (self.model is not None and
                    self.probabilistic_model is not None and
                    hasattr(self.model, 'predict') and
                    hasattr(self.probabilistic_model, 'predict')):

                # Проверяем возможность дообучения
                if hasattr(self.model, 'warm_start'):
                    print(f"   ✓ RandomForest поддерживает дообучение: {self.model.warm_start}")

                # Проверяем scaler
                if hasattr(self.scaler, 'n_features_in_'):
                    print(f"   ✓ Scaler настроен на {self.scaler.n_features_in_} признаков")

                return True
            return False
        except Exception as e:
            print(f"❌ Ошибка валидации моделей: {e}")
            return False

    def _create_new_models(self):
        """Создание новых моделей с поддержкой дообучения"""
        try:
            # RandomForest с включенным warm_start для дообучения
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
                verbose=0,
                warm_start=True  # Ключевой параметр для дообучения
            )

            # BayesianRidge также поддерживает дообучение
            self.probabilistic_model = BayesianRidge(
                n_iter=300,
                tol=1e-3,
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6
            )

            # Инициализируем метаданные
            self.model_metadata = {
                'creation_date': datetime.now().isoformat(),
                'total_training_samples': 0,
                'training_sessions': 0,
                'last_training': None,
                'model_type': 'EnhancedStockAIModel_v3',
                'feature_count': 10,
                'warm_start_enabled': True
            }

            self.is_trained = False
            print("✅ Новые модели созданы с поддержкой дообучения")
            return True

        except Exception as e:
            print(f"❌ Ошибка создания моделей: {e}")
            return False

    def can_do_incremental_learning(self):
        """Проверка возможности дообучения модели"""
        if not self.is_trained:
            return False

        checks = {
            'models_exist': self.model is not None and self.probabilistic_model is not None,
            'models_trained': self.is_trained,
            'warm_start': hasattr(self.model, 'warm_start') and self.model.warm_start,
            'scaler_ready': hasattr(self.scaler, 'n_features_in_') or not hasattr(self.scaler, 'mean_')
        }

        return all(checks.values())

    def train_models(self, prices_df, news_sentiment_by_date, incremental=True):
        """Улучшенное обучение моделей с поддержкой дообучения"""
        try:
            print("🔄 Запуск обучения моделей...")

            # Проверяем и создаем модели при необходимости
            if self.model is None or self.probabilistic_model is None:
                print("🔧 Модели не инициализированы, создаем новые...")
                self._create_new_models()

            # Подготавливаем данные
            X, y = self.prepare_training_data(prices_df, news_sentiment_by_date)

            if len(X) == 0 or len(y) == 0:
                print("❌ Недостаточно данных для обучения")
                return False

            print(f"🎯 Обучение на {len(X)} примерах...")

            # Определяем режим обучения
            can_incremental = self.can_do_incremental_learning() and incremental

            if can_incremental:
                print("🔁 Режим дообучения существующей модели...")
                X_processed = self._prepare_for_incremental_learning(X)

                # Дообучаем модели
                self.model.fit(X_processed, y)
                self.probabilistic_model.fit(X_processed, y)

                training_type = "incremental"
            else:
                print("🆕 Режим полного обучения...")
                X_processed = self.scaler.fit_transform(X)
                self.model.fit(X_processed, y)
                self.probabilistic_model.fit(X_processed, y)
                training_type = "full"

            # Оцениваем качество
            train_score = self.model.score(X_processed, y)
            self.is_trained = True

            # Обновляем метаданные
            self._update_training_metadata(len(X), train_score, training_type)

            # Сохраняем модели
            self._save_models()

            print(f"✅ Обучение завершено ({training_type}), качество: {train_score:.4f}")
            return True

        except Exception as e:
            print(f"❌ Ошибка обучения моделей: {e}")
            traceback.print_exc()
            return False

    def _prepare_for_incremental_learning(self, X_new):
        """Подготовка данных для инкрементального обучения"""
        try:
            # Проверяем совместимость данных с существующим scaler
            if (hasattr(self.scaler, 'n_features_in_') and
                    X_new.shape[1] == self.scaler.n_features_in_):
                return self.scaler.transform(X_new)
            else:
                print("⚠️ Несовместимость данных, создаем новый scaler")
                self.scaler = StandardScaler()
                return self.scaler.fit_transform(X_new)

        except Exception as e:
            print(f"⚠️ Ошибка подготовки данных: {e}, создаем новый scaler")
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X_new)

    def _update_training_metadata(self, new_samples_count, train_score, training_type):
        """Обновление метаданных обучения"""
        # Обновляем общую статистику
        current_total = self.model_metadata.get('total_training_samples', 0)
        self.model_metadata['total_training_samples'] = current_total + new_samples_count
        self.model_metadata['training_sessions'] = self.model_metadata.get('training_sessions', 0) + 1
        self.model_metadata['last_training'] = datetime.now().isoformat()
        self.model_metadata['last_training_type'] = training_type

        # Сохраняем запись в истории
        training_record = {
            'timestamp': datetime.now(),
            'new_samples': new_samples_count,
            'total_samples': self.model_metadata['total_training_samples'],
            'train_score': train_score,
            'training_type': training_type,
            'session_number': self.model_metadata['training_sessions']
        }
        self.training_history.append(training_record)

        print(f"📊 Метаданные обновлены: всего {self.model_metadata['total_training_samples']} примеров")

    def _save_models(self):
        """Сохранение моделей со всей метаинформацией"""
        try:
            models_to_save = {
                'main_model': self.model,
                'probabilistic_model': self.probabilistic_model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'model_metadata': self.model_metadata
            }

            joblib.dump(models_to_save, self.model_path)
            print(f"💾 Модели успешно сохранены в {self.model_path}")

        except Exception as e:
            print(f"❌ Ошибка сохранения моделей: {e}")

    def load_or_create_model(self):
        """Публичный метод для загрузки или создания модели"""
        return self._load_model_safe() or self._create_new_models()

    # Остальные методы остаются без изменений
    def create_advanced_features(self, prices_df, news_sentiment):
        """Создание расширенных признаков с улучшенной обработкой ошибок"""
        features = {}

        if prices_df.empty:
            print("⚠️ Нет данных для создания признаков")
            return features

        print(f"🔍 Создание признаков для {len(prices_df.columns)} тикеров...")

        for ticker in prices_df.columns:
            try:
                if ticker not in prices_df:
                    continue

                price_series = prices_df[ticker].dropna()

                # Увеличиваем минимальное требование для более стабильных признаков
                if len(price_series) < 30:
                    continue

                # Базовые статистики с улучшенной обработкой
                returns = price_series.pct_change().dropna()

                if len(returns) < 15:
                    continue

                try:
                    # Волатильность с разными периодами
                    volatility_short = returns.rolling(window=10, min_periods=8).std().iloc[-1]
                    volatility_long = returns.rolling(window=20, min_periods=15).std().iloc[-1]

                    # Скользящие средние
                    ma_short = price_series.rolling(window=5, min_periods=4).mean()
                    ma_medium = price_series.rolling(window=10, min_periods=8).mean()
                    ma_long = price_series.rolling(window=20, min_periods=15).mean()

                    # Отношения скользящих средних
                    ma_ratio_short = ma_short.iloc[-1] / ma_long.iloc[-1] if ma_long.iloc[-1] != 0 else 1.0
                    ma_ratio_medium = ma_medium.iloc[-1] / ma_long.iloc[-1] if ma_long.iloc[-1] != 0 else 1.0

                    # RSI с улучшенным расчетом
                    delta = price_series.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=10).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=10).mean()

                    rsi = 50.0
                    if not (pd.isna(loss.iloc[-1]) or loss.iloc[-1] == 0):
                        rs = gain.iloc[-1] / loss.iloc[-1]
                        rsi = 100 - (100 / (1 + rs))
                    rsi = max(1.0, min(99.0, rsi))  # Более строгие границы

                    # Моментум разных периодов
                    momentum_5 = (price_series.iloc[-1] / price_series.iloc[-5] - 1) if len(price_series) >= 5 else 0
                    momentum_10 = (price_series.iloc[-1] / price_series.iloc[-10] - 1) if len(price_series) >= 10 else 0

                    # Процент от максимума/минимума
                    price_max_20 = price_series.rolling(window=20, min_periods=15).max().iloc[-1]
                    price_min_20 = price_series.rolling(window=20, min_periods=15).min().iloc[-1]
                    high_low_ratio = (price_series.iloc[-1] - price_min_20) / (price_max_20 - price_min_20) if (
                                                                                                                           price_max_20 - price_min_20) != 0 else 0.5

                    # Сентимент сектора
                    sector_sentiment = news_sentiment.get(self.get_sector_for_ticker(ticker), 0)

                    features[ticker] = {
                        'volatility_short': float(volatility_short) if not pd.isna(volatility_short) else 0.02,
                        'volatility_long': float(volatility_long) if not pd.isna(volatility_long) else 0.02,
                        'ma_ratio_short': float(ma_ratio_short),
                        'ma_ratio_medium': float(ma_ratio_medium),
                        'rsi': float(rsi),
                        'momentum_5': float(momentum_5),
                        'momentum_10': float(momentum_10),
                        'high_low_ratio': float(high_low_ratio),
                        'sector_sentiment': float(sector_sentiment),
                        'volume_stability': float(np.log1p(len(price_series)))  # Стабильность данных
                    }

                except Exception as calc_error:
                    print(f"⚠️ Ошибка расчета признаков для {ticker}: {calc_error}")
                    continue

            except Exception as e:
                print(f"⚠️ Общая ошибка обработки {ticker}: {e}")
                continue

        print(f"📊 Создано признаков для {len(features)} тикеров")
        return features

    def get_sector_for_ticker(self, ticker):
        """Определение сектора для тикера"""
        for sector, tickers in RUSSIAN_SECTORS.items():
            if ticker in tickers:
                return sector
        return 'Общие'

    def prepare_training_data(self, prices_df, news_sentiment_by_date, forecast_days=5):
        """Улучшенная подготовка данных для обучения"""
        X, y = [], []

        if prices_df.empty:
            print("⚠️ Нет данных для обучения")
            return np.array([]), np.array([])

        # Создаем базовый сентимент если нет данных
        if not news_sentiment_by_date:
            base_sentiment = {sector: 0 for sector in RUSSIAN_SECTORS.keys()}
            news_sentiment_by_date = {datetime.now().date(): base_sentiment}

        total_samples = 0
        successful_tickers = 0

        print("📚 Подготовка данных для обучения...")

        for ticker in prices_df.columns:
            try:
                price_data = prices_df[ticker].dropna()

                # Увеличиваем минимальные требования для качественного обучения
                if len(price_data) < 50:
                    continue

                # Используем последний доступный сентимент
                current_sentiment = {}
                if news_sentiment_by_date:
                    latest_date = max(news_sentiment_by_date.keys())
                    current_sentiment = news_sentiment_by_date[latest_date]

                ticker_samples = 0

                # Создаем признаки для каждого окна с улучшенной логикой
                for i in range(30, len(price_data) - forecast_days):
                    try:
                        historical_segment = price_data.iloc[:i + 1]

                        # Проверяем достаточность исторических данных
                        if len(historical_segment) < 30:
                            continue

                        # Целевая переменная - доходность через forecast_days дней
                        future_price = price_data.iloc[i + forecast_days]
                        current_price = price_data.iloc[i]

                        if current_price <= 0:
                            continue

                        target_return = (future_price - current_price) / current_price

                        # Ограничиваем экстремальные значения
                        target_return = np.clip(target_return, -0.5, 0.5)

                        # Создаем признаки для исторического сегмента
                        features_dict = self.create_advanced_features(
                            pd.DataFrame({ticker: historical_segment}),
                            current_sentiment
                        )

                        if ticker in features_dict and features_dict[ticker]:
                            feature_vector = list(features_dict[ticker].values())

                            # Улучшенная проверка валидности признаков
                            if (len(feature_vector) == 10 and
                                    all(not pd.isna(x) for x in feature_vector) and
                                    all(np.isfinite(x) for x in feature_vector) and
                                    all(abs(x) < 100 for x in feature_vector)):  # Защита от выбросов

                                X.append(feature_vector)
                                y.append(target_return)
                                total_samples += 1
                                ticker_samples += 1

                    except Exception as window_error:
                        continue

                if ticker_samples > 0:
                    successful_tickers += 1
                    print(f"   {ticker}: {ticker_samples} примеров")

            except Exception as e:
                print(f"⚠️ Ошибка подготовки данных для {ticker}: {e}")
                continue

        print(f"✅ Подготовлено {len(X)} примеров для обучения с {successful_tickers} тикеров")

        if len(X) > 0:
            X_array = np.array(X, dtype=np.float64)
            y_array = np.array(y, dtype=np.float64)
            return X_array, y_array
        else:
            return np.array([]), np.array([])

    def predict_with_confidence(self, current_features):
        """Улучшенный прогноз с оценкой достоверности"""
        if not self.is_trained or self.model is None:
            print("⚠️ Модель не обучена, возвращаем нулевой прогноз")
            return 0.0, 1.0  # Высокая неопределенность

        try:
            # Проверяем и преобразуем признаки
            if isinstance(current_features, dict):
                feature_vector = list(current_features.values())
            else:
                feature_vector = current_features

            # Улучшенная проверка валидности признаков
            if (len(feature_vector) != 10 or
                    any(pd.isna(x) for x in feature_vector) or
                    any(not np.isfinite(x) for x in feature_vector)):
                print("⚠️ Невалидные признаки для прогноза")
                return 0.0, 1.0

            # Преобразуем и нормализуем признаки
            features_array = np.array(feature_vector).reshape(1, -1)

            try:
                features_array = self.scaler.transform(features_array)
            except Exception as scale_error:
                print(f"⚠️ Ошибка нормализации признаков: {scale_error}")
                return 0.0, 1.0

            # Прогноз основной модели
            main_pred = self.model.predict(features_array)[0]

            # Прогноз вероятностной модели
            probabilistic_pred, probabilistic_std = self.probabilistic_model.predict(features_array, return_std=True)

            # Комбинируем прогнозы с весами
            combined_pred = 0.7 * main_pred + 0.3 * probabilistic_pred

            # Ограничиваем экстремальные прогнозы
            combined_pred = np.clip(combined_pred, -0.2, 0.2)

            # Рассчитываем итоговую достоверность
            confidence = max(0.0, min(1.0, 1.0 - probabilistic_std[0]))

            return float(combined_pred), float(confidence)

        except Exception as e:
            print(f"❌ Ошибка прогнозирования: {e}")
            return 0.0, 0.0  # Нулевая достоверность при ошибке

    def get_training_info(self):
        """Получение информации о тренировке модели"""
        if not self.training_history:
            return "Модель еще не обучалась"

        latest = self.training_history[-1]
        return {
            'last_training': latest['timestamp'],
            'samples': latest['total_samples'],
            'train_score': latest['train_score'],
            'training_type': latest['training_type'],
            'total_trainings': len(self.training_history),
            'can_incremental_learn': self.can_do_incremental_learning()
        }

# ==================== ОСНОВНОЙ КЛАСС СИСТЕМЫ С ИСПРАВЛЕНИЯМИ ====================
class EnhancedStockAnalysisSystem:
    def __init__(self):
        # Эти классы должны быть определены выше в коде
        self.news_collector = NewsCollector()
        self.stock_collector = StockDataCollector()
        self.news_analyzer = NewsAnalyzer()
        self.enhanced_ai_model = EnhancedStockAIModel()
        self.portfolio = VirtualPortfolio()
        self.portfolio_rebalancer = PortfolioRebalancer(
            self.portfolio,
            threshold=0.05,
            strategy='calendar_absolute'
        )
        self.web_interface = WebInterface()

        self.news_backup_file = 'news_backup.csv'
        self.prices_backup_file = 'prices_backup.csv'
        self.recommendations_file = 'RECOM.csv'

    def run_enhanced_analysis_cycle(self):
        """Улучшенный цикл анализа с приоритизацией обучения ИИ"""
        print("\n" + "=" * 60)
        print(f"🔍 Запуск УЛУЧШЕННОГО анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        try:
            # 1. Сбор данных
            print("📰 Шаг 1: Сбор новостей...")
            news_df = self.news_collector.fetch_all_news()

            print("📊 Шаг 2: Сбор данных об акциях...")
            all_tickers = [ticker for sector_tickers in RUSSIAN_SECTORS.values() for ticker in sector_tickers]
            prices_df = self.stock_collector.get_stock_prices(all_tickers)

            # 2. Базовый анализ новостей
            print("🔍 Шаг 3: Анализ новостей...")
            news_recommendations = self.news_analyzer.predict_sector_sentiment(news_df)

            # 3. Обучение моделей - ТЕПЕРЬ С ПРИОРИТЕТОМ
            print("🎯 Шаг 4: Обучение AI моделей...")
            news_sentiment_by_date = {datetime.now().date(): news_recommendations}

            # Всегда пытаемся обучить модель, если есть достаточно данных
            training_successful = self.enhanced_ai_model.train_models(prices_df, news_sentiment_by_date)

            if not training_successful:
                print("⚠️ Обучение не удалось, пробуем загрузить существующую модель...")
                model_loaded = self.enhanced_ai_model.load_or_create_model()
                if not model_loaded:
                    print("❌ Не удалось ни обучить, ни загрузить модель")
            else:
                print("✅ Модели успешно обучены на новых данных")

            # 4. Расширенный AI анализ
            print("🧠 Шаг 5: Расширенный AI анализ...")
            current_prices = prices_df.iloc[-1].to_dict() if not prices_df.empty else {}
            advanced_features = self.enhanced_ai_model.create_advanced_features(prices_df, news_recommendations)

            ai_predictions = {}
            for ticker, features in advanced_features.items():
                if features:
                    prediction, confidence = self.enhanced_ai_model.predict_with_confidence(features)
                    ai_predictions[ticker] = {
                        'predicted_return': prediction,
                        'confidence': confidence,
                        'sector': self.enhanced_ai_model.get_sector_for_ticker(ticker)
                    }

            # 5. Ребалансировка портфеля
            print("⚖️ Шаг 6: Проверка ребалансировки портфеля...")
            rebalancing_trades = self.portfolio_rebalancer.execute_rebalancing(current_prices, news_recommendations)

            # 6. Улучшенные рекомендации
            enhanced_recommendations = self._enhance_recommendations_with_ai(news_recommendations, ai_predictions)

            # 7. Сохранение и отображение
            portfolio_value = self.portfolio.get_portfolio_value(current_prices)
            self.save_backup_data(news_df, prices_df, enhanced_recommendations)
            self.web_interface.update_data(enhanced_recommendations, news_df, portfolio_value)

            # 8. Вывод результатов с информацией о модели
            self._print_enhanced_results(enhanced_recommendations, ai_predictions, rebalancing_trades, portfolio_value)

            return True

        except Exception as e:
            print(f"💥 Критическая ошибка в цикле анализа: {e}")
            traceback.print_exc()
            return False

    def _enhance_recommendations_with_ai(self, news_recommendations, ai_predictions):
        """Улучшение рекомендаций с помощью AI прогнозов"""
        enhanced_recommendations = news_recommendations.copy()

        sector_predictions = {}
        for ticker_data in ai_predictions.values():
            sector = ticker_data['sector']
            if sector not in sector_predictions:
                sector_predictions[sector] = []

            # Учитываем достоверность прогноза
            weighted_prediction = ticker_data['predicted_return'] * ticker_data['confidence']
            sector_predictions[sector].append(weighted_prediction)

        for sector, predictions in sector_predictions.items():
            if predictions:
                avg_prediction = np.mean(predictions)
                # Более агрессивное преобразование в рейтинг
                ai_adjustment = int(round(avg_prediction * 200))  # Увеличили коэффициент
                ai_adjustment = max(-5, min(5, ai_adjustment))

                if sector in enhanced_recommendations:
                    # Взвешенное объединение с новостными рекомендациями
                    news_weight = 0.3
                    ai_weight = 0.7
                    enhanced_recommendations[sector] = int(
                        news_weight * enhanced_recommendations[sector] + ai_weight * ai_adjustment
                    )
                else:
                    enhanced_recommendations[sector] = ai_adjustment

        return enhanced_recommendations

    def _print_enhanced_results(self, recommendations, ai_predictions, trades, portfolio_value):
        """Вывод расширенных результатов анализа с информацией о модели"""
        print("\n📈 УЛУЧШЕННЫЕ РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print("-" * 60)

        # Информация о модели
        model_info = self.enhanced_ai_model.get_training_info()
        print(f"🤖 ИНФОРМАЦИЯ О МОДЕЛИ:")
        if isinstance(model_info, dict):
            print(f"   Последнее обучение: {model_info['last_training']}")
            print(f"   Примеров для обучения: {model_info['samples']}")
            print(f"   Качество (R²): {model_info.get('test_score', model_info.get('train_score', 0.0)):.4f}")
            print(f"   Всего обучений: {model_info['total_trainings']}")
        else:
            print(f"   {model_info}")

        print("\n🎯 РЕКОМЕНДАЦИИ ПО СЕКТОРАМ:")
        for sector, rec in sorted(recommendations.items(), key=lambda x: x[1], reverse=True):
            emoji = "🟢" if rec > 0 else "🔴" if rec < 0 else "⚪"
            print(f"{emoji} {sector:30} {rec:+2d}/5")

        print(f"\n🤖 AI ПРОГНОЗЫ (топ-5 по достоверности):")
        # Сортируем по достоверности
        sorted_predictions = sorted(ai_predictions.items(),
                                    key=lambda x: x[1]['confidence'], reverse=True)[:5]

        for ticker, data in sorted_predictions:
            return_pct = data['predicted_return'] * 100
            confidence = data['confidence'] * 100
            print(f"   {ticker:6} | {return_pct:+.1f}% | достоверность: {confidence:.1f}%")

        if trades:
            print(f"\n⚖️ ВЫПОЛНЕННЫЕ СДЕЛКИ: {len(trades)}")
            for trade in trades:
                print(f"   {trade['action']} {trade['ticker']} {trade['quantity']} шт. - {trade['reason']}")

        formatted_value = f"₽{portfolio_value:,.2f}"
        print(f"\n💼 ТЕКУЩАЯ СТОИМОСТЬ ПОРТФЕЛЯ: {formatted_value}")
        print("✅ УЛУЧШЕННЫЙ АНАЛИЗ ЗАВЕРШЕН!")

    def save_backup_data(self, news_df, prices_df, recommendations):
        """Сохранение резервных копий данных"""
        try:
            if not news_df.empty:
                news_df.to_csv(self.news_backup_file, index=False, encoding='utf-8-sig')

            if not prices_df.empty:
                prices_df.to_csv(self.prices_backup_file, index=False, encoding='utf-8-sig')

            rec_df = pd.DataFrame(list(recommendations.items()), columns=['Сектор', 'Рекомендация'])
            rec_df['Дата'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rec_df['Описание'] = rec_df['Рекомендация'].apply(
                lambda x: 'Максимальная покупка' if x == 5 else
                'Сильная покупка' if x >= 3 else
                'Умеренная покупка' if x >= 1 else
                'Нейтрально' if x == 0 else
                'Умеренная продажа' if x >= -2 else
                'Сильная продажа' if x >= -4 else
                'Максимальная продажа'
            )
            rec_df.to_csv(self.recommendations_file, index=False, encoding='utf-8-sig')

            print("💾 Данные сохранены в резервные файлы")

        except Exception as e:
            print(f"Ошибка сохранения резервных копий: {e}")

    def run_continuous_analysis(self, interval_minutes=30):
        """Запуск непрерывного анализа"""
        print("🚀 Запуск УЛУЧШЕННОЙ системы анализа российских акций с ИИ...")
        print("🔧 Возможности: Улучшенные модели + Расширенные признаки + Профессиональная валидация")

        web_thread = threading.Thread(target=self.web_interface.run_server)
        web_thread.daemon = True
        web_thread.start()

        cycle_count = 0
        while self.web_interface.is_running:
            try:
                success = self.run_enhanced_analysis_cycle()
                cycle_count += 1

                if success:
                    print(f"\n♻️ Цикл {cycle_count} завершен. Ожидание следующего цикла...")

                    # Ждем указанное количество минут
                    wait_seconds = interval_minutes * 60
                    for i in range(wait_seconds):
                        if not self.web_interface.is_running:
                            break
                        time.sleep(1)
                else:
                    print("💤 Ожидание 5 минут перед повторной попыткой...")
                    time.sleep(300)

            except KeyboardInterrupt:
                print("\n⚠️ Программа прервана пользователем")
                break
            except Exception as e:
                print(f"💥 Критическая ошибка: {e}")
                traceback.print_exc()
                time.sleep(300)

        print("🛑 Программа остановлена")


# ==================== ЗАПУСК ПРОГРАММЫ ====================
if __name__ == "__main__":
    system = EnhancedStockAnalysisSystem()
    system.run_continuous_analysis(interval_minutes=3)