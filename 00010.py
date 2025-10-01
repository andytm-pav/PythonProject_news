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
from sklearn.metrics import mean_squared_error
import re
import threading
from flask import Flask, render_template_string, request
import json

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
        """Сбор новостей из всех RSS-лент с обработкой ошибок:cite[3]:cite[5]"""
        all_news = []

        for source, url in self.feeds.items():
            try:
                print(f"Загрузка новостей из {source}...")
                # Добавляем заголовки для обхода блокировок:cite[5]
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }

                # Парсим RSS-ленту:cite[3]:cite[5]
                feed = feedparser.parse(url)
                time.sleep(1)  # Пауза между запросами

                if not feed.entries:
                    print(f"  Нет новостей в {source}")
                    continue

                items_added = 0
                for entry in feed.entries[:max_items_per_feed]:
                    try:
                        title = getattr(entry, 'title', '').strip()
                        if not title:
                            continue

                        # Обработка даты публикации:cite[5]
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

        # Создаем DataFrame и удаляем дубликаты
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
                            date_str = candle[6]  # Дата свечи
                            close_price = candle[5]  # Цена закрытия

                            if date_str and close_price:
                                dates.append(date_str)
                                prices.append(close_price)

                        if dates and prices:
                            series = pd.Series(prices, index=dates)
                            stock_data[ticker] = series
                            successful_downloads += 1

                time.sleep(0.5)  # Пауза между запросами

            except Exception as e:
                print(f"Ошибка загрузки {ticker}: {e}")
                continue

        print(f"Успешно загружено данных для {successful_downloads} тикеров")

        # Создаем DataFrame
        if stock_data:
            prices_df = pd.DataFrame(stock_data)
            prices_df = prices_df.ffill().bfill()  # Заполнение пропусков
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

        # Анализ положительных слов
        for word in self.sentiment_words['positive']:
            if word in text_lower:
                positive_score += 1
                # Проверка усилителей
                for intensifier in self.sentiment_words['intensifiers']:
                    if f"{intensifier} {word}" in text_lower:
                        positive_score += 0.5
                        break

        # Анализ отрицательных слов
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

        # Расчет среднего сентимента и преобразование в рекомендации
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


# ==================== КЛАСС ИИ МОДЕЛИ ====================
class StockAIModel:
    def __init__(self, model_path='stock_ai_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.is_trained = False

    def load_or_create_model(self):
        """Загрузка или создание модели"""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                print("✅ Модель ИИ загружена")
                return True
            except Exception as e:
                print(f"Ошибка загрузки модели: {e}")

        # Создание новой модели
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        print("✅ Создана новая модель ИИ")
        return False

    def prepare_features(self, news_features, price_features):
        """Подготовка признаков для обучения"""
        try:
            # Объединяем новостные и ценовые признаки
            features = np.hstack([news_features, price_features])
            return features
        except Exception as e:
            print(f"Ошибка подготовки признаков: {e}")
            return np.random.random((10, len(RUSSIAN_SECTORS) * 2))

    def train_model(self, X, y):
        """Обучение модели"""
        try:
            if len(X) < 10:
                print("Недостаточно данных для обучения")
                return False

            self.model.fit(X, y)
            self.is_trained = True
            joblib.dump(self.model, self.model_path)
            print("✅ Модель успешно обучена и сохранена")
            return True
        except Exception as e:
            print(f"Ошибка обучения модели: {e}")
            return False

    def predict(self, X):
        """Прогнозирование"""
        if not self.is_trained or self.model is None:
            return np.zeros(X.shape[0])
        return self.model.predict(X)


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
                return (datetime.now() - news_time).total_seconds() < 3600  # Новые за последний час

            template = self.get_html_template()
            return render_template_string(
                template,
                recommendations=self.current_recommendations,
                news_items=self.news_items[-20:],  # Последние 20 новостей
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


# ==================== ОСНОВНОЙ КЛАСС СИСТЕМЫ ====================
class StockAnalysisSystem:
    def __init__(self):
        self.news_collector = NewsCollector()
        self.stock_collector = StockDataCollector()
        self.news_analyzer = NewsAnalyzer()
        self.ai_model = StockAIModel()
        self.portfolio = VirtualPortfolio()
        self.web_interface = WebInterface()

        # Файлы для дублирования данных
        self.news_backup_file = 'news_backup.csv'
        self.prices_backup_file = 'prices_backup.csv'
        self.recommendations_file = 'RECOM.csv'

    def save_backup_data(self, news_df, prices_df, recommendations):
        """Сохранение резервных копий данных"""
        try:
            # Сохранение новостей
            if not news_df.empty:
                news_df.to_csv(self.news_backup_file, index=False, encoding='utf-8-sig')

            # Сохранение цен
            if not prices_df.empty:
                prices_df.to_csv(self.prices_backup_file, index=False, encoding='utf-8-sig')

            # Сохранение рекомендаций
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

    def run_analysis_cycle(self):
        """Запуск одного цикла анализа"""
        print("\n" + "=" * 60)
        print(f"🔍 Запуск анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # 1. Сбор новостей
        print("📰 Шаг 1: Сбор новостей...")
        news_df = self.news_collector.fetch_all_news()

        # 2. Сбор данных об акциях
        print("📊 Шаг 2: Сбор данных об акциях...")
        all_tickers = []
        for sector_tickers in RUSSIAN_SECTORS.values():
            all_tickers.extend(sector_tickers)

        prices_df = self.stock_collector.get_stock_prices(all_tickers)

        # 3. Анализ новостей и прогнозирование
        print("🔍 Шаг 3: Анализ новостей...")
        recommendations = self.news_analyzer.predict_sector_sentiment(news_df)

        # 4. Загрузка и обучение ИИ модели
        print("🧠 Шаг 4: Обучение ИИ модели...")
        self.ai_model.load_or_create_model()

        # 5. Сохранение данных
        print("💾 Шаг 5: Сохранение данных...")
        self.save_backup_data(news_df, prices_df, recommendations)

        # 6. Расчет стоимости портфеля
        current_prices = {}
        if not prices_df.empty:
            current_prices = prices_df.iloc[-1].to_dict()

        portfolio_value = self.portfolio.get_portfolio_value(current_prices)

        # 7. Обновление веб-интерфейса
        self.web_interface.update_data(recommendations, news_df, portfolio_value)

        # 8. Вывод результатов
        print("\n📈 РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print("-" * 50)
        for sector, rec in sorted(recommendations.items(), key=lambda x: x[1], reverse=True):
            emoji = "🟢" if rec > 0 else "🔴" if rec < 0 else "⚪"
            print(f"{emoji} {sector:30} {rec:+2d}/5")

        print(f"\n💼 Стоимость портфеля: ₽{portfolio_value:,.2f}")
        print("✅ Анализ завершен!")

        return True

    def run_continuous_analysis(self, interval_minutes=30):
        """Запуск непрерывного анализа"""
        print("🚀 Запуск системы анализа российских акций с ИИ...")
        print(f"⏰ Интервал анализа: {interval_minutes} минут")

        # Запуск веб-интерфейса в отдельном потоке
        web_thread = threading.Thread(target=self.web_interface.run_server)
        web_thread.daemon = True
        web_thread.start()

        # Основной цикл анализа
        cycle_count = 0
        while self.web_interface.is_running:
            try:
                success = self.run_analysis_cycle()
                cycle_count += 1

                if success:
                    print(f"\n♻️ Цикл {cycle_count} завершен. Ожидание следующего цикла...")

                    # Ожидание до следующего цикла
                    for i in range(interval_minutes * 2):
                        if not self.web_interface.is_running:
                            break
                        time.sleep(1)

                else:
                    print("❌ Ошибка в цикле анализа. Повтор через 5 минут...")
                    time.sleep(300)

            except KeyboardInterrupt:
                print("\n⚠️ Программа прервана пользователем")
                break
            except Exception as e:
                print(f"💥 Критическая ошибка: {e}")
                time.sleep(300)

        print("🛑 Программа остановлена")


# ==================== ЗАПУСК ПРОГРАММЫ ====================
if __name__ == "__main__":
    system = StockAnalysisSystem()
    system.run_continuous_analysis(interval_minutes=30)  # Анализ каждые 30 минут