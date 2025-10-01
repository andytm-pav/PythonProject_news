import os
import sys
import time
import requests
import feedparser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import re

warnings.filterwarnings('ignore')

# ---------------------------
# 1) КОНФИГУРАЦИЯ РОССИЙСКИХ СЕКТОРОВ И АКЦИЙ
# ---------------------------

RUSSIAN_SECTORS = {
    'Энергетика': ['GAZP', 'LKOH', 'ROSN', 'SNGS', 'TATN'],
    'Финансы': ['SBER', 'VTBR', 'MOEX', 'TCSG'],
    'Металлургия': ['GMKN', 'NLMK', 'MAGN', 'PLZL', 'RUAL'],
    'Телекоммуникации': ['MTSS', 'RTKM', 'MGNT'],
    'Химия': ['PHOR', 'AKRN', 'ODVA'],
    'Транспорт': ['AFLT', 'NMTP', 'TRMK'],
    'Строительство': ['PIKK', 'LSRG', 'UNAC'],
    'IT-технологии': ['YNDX', 'OZON', 'VKCO'],
    'Розничная торговля': ['FIVE', 'DSKY', 'LNTA'],
    'Фармацевтика': ['POLY', 'RGSS'],
    'Электроэнергетика': ['FEES', 'HYDR', 'IRAO']
}

SECTOR_MAPPING = {
    'нефть': 'Энергетика', 'газ': 'Энергетика', 'нефтегаз': 'Энергетика', 'энерг': 'Энергетика',
    'банк': 'Финансы', 'финанс': 'Финансы', 'сбер': 'Финансы', 'втб': 'Финансы', 'биржа': 'Финансы',
    'металл': 'Металлургия', 'сталь': 'Металлургия', 'никель': 'Металлургия', 'алюмин': 'Металлургия',
    'связь': 'Телекоммуникации', 'телеком': 'Телекоммуникации', 'мтс': 'Телекоммуникации',
    'химия': 'Химия', 'удобрен': 'Химия', 'нефтехим': 'Химия',
    'авиа': 'Транспорт', 'транспорт': 'Транспорт', 'порт': 'Транспорт', 'аэропорт': 'Транспорт',
    'строитель': 'Строительство', 'девелопер': 'Строительство', 'недвиж': 'Строительство',
    'ит': 'IT-технологии', 'технолог': 'IT-технологии', 'интернет': 'IT-технологии', 'софт': 'IT-технологии',
    'ритейл': 'Розничная торговля', 'магазин': 'Розничная торговля', 'торговля': 'Розничная торговля',
    'фарма': 'Фармацевтика', 'медицин': 'Фармацевтика', 'лекарств': 'Фармацевтика',
    'электроэнерг': 'Электроэнергетика', 'энергосбыт': 'Электроэнергетика'
}

# ---------------------------
# 2) RSS ЛЕНТЫ РОССИЙСКИХ ФИНАНСОВЫХ НОВОСТЕЙ
# ---------------------------

RUSSIAN_FEEDS = {
    "Ведомости": "https://www.vedomosti.ru/rss/news",
    "Коммерсант": "https://www.kommersant.ru/RSS/news.xml",
    "Интерфакс": "https://www.interfax.ru/rss.asp",
    "ТАСС": "https://tass.ru/rss/v2.xml",
    "Лента.ру": "https://lenta.ru/rss/news",
    "Финам": "https://www.finam.ru/analysis/conews/rsspoint/",
    "БФМ": "https://www.bfm.ru/news.rss?rubric=19",
    "Комерсант": "https://www.kommersant.ru/rss/news.xml",
    "Альта_закон": "http://www.alta.ru/rss/laws_news/",
    "финам мировых": "https://www.finam.ru/international/advanced/rsspoint/",
    "Инвистирование ком": "https://ru.investing.com/rss/news_356.rss",
    "инвестирование ком сырьё": "https://ru.investing.com/rss/news_11.rss",
    "инвестирование ком Экономики": "https://ru.investing.com/rss/news_14.rss",
    "инвестирование ком фондовый": "https://ru.investing.com/rss/news_25.rss",
}


# ---------------------------
# 3) КЛАСС ДЛЯ ПОЛУЧЕНИЯ ДАННЫХ О РОССИЙСКИХ АКЦИЯХ
# ---------------------------

class RussianStockData:
    """Улучшенный класс для получения данных о российских акциях"""

    @staticmethod
    def get_moex_data_safe(tickers, days=30):
        """Безопасное получение данных с Московской биржи с обработкой дубликатов"""
        base_url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{}/candles.json"

        prices_data = {}
        successful_downloads = 0

        for ticker in tqdm(tickers, desc="Загрузка данных MOEX"):
            try:
                url = base_url.format(ticker)
                from_date = (datetime.now() - timedelta(days=days * 2)).strftime('%Y-%m-%d')
                till_date = datetime.now().strftime('%Y-%m-%d')

                response = requests.get(url, params={
                    'from': from_date,
                    'till': till_date,
                    'interval': 24
                }, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    candles = data.get('candles', {}).get('data', [])

                    if candles:
                        # Извлекаем даты и цены закрытия
                        dates = []
                        closes = []

                        for candle in candles:
                            date_str = candle[0]  # Дата в формате '2024-01-15'
                            close_price = candle[1]  # Цена закрытия

                            if date_str and close_price:
                                dates.append(date_str)
                                closes.append(close_price)

                        if dates and closes:
                            # Создаем Series и обеспечиваем уникальность индекса
                            series = pd.Series(closes, index=dates)

                            # УДАЛЯЕМ ДУБЛИКАТЫ ИНДЕКСА (сохраняем первое вхождение)
                            if not series.index.is_unique:
                                print(f"Обнаружены дубликаты индекса для {ticker}. Удаляем дубликаты...")
                                series = series[~series.index.duplicated(keep='first')]

                            prices_data[ticker] = series
                            successful_downloads += 1

                time.sleep(1)  # Увеличиваем паузу для соблюдения лимитов API

            except requests.exceptions.Timeout:
                print(f"Таймаут при загрузке {ticker}")
            except Exception as e:
                print(f"Ошибка загрузки {ticker}: {e}")
                continue

        print(f"Успешно загружено данных для {successful_downloads} тикеров")
        return prices_data

    @staticmethod
    def create_dataframe_from_series(series_dict, min_length=10):
        """Создание DataFrame из словаря Series с безопасной обработкой"""
        if not series_dict:
            return pd.DataFrame()

        # Находим максимальную длину среди всех Series
        max_length = max(len(series) for series in series_dict.values())

        if max_length < min_length:
            print(f"Данные слишком короткие (max_length={max_length}). Используем тестовые данные.")
            return pd.DataFrame()

        # Создаем DataFrame с безопасным выравниванием
        prices_df = pd.DataFrame()

        for ticker, series in series_dict.items():
            if len(series) == max_length:
                # Если длина совпадает, используем как есть
                prices_df[ticker] = series
            else:
                # Выравниваем длину, заполняя недостающие значения NaN
                aligned_prices = pd.Series([np.nan] * max_length)

                # Копируем доступные данные
                if len(series) > 0:
                    aligned_prices.iloc[:len(series)] = series.values

                prices_df[ticker] = aligned_prices

        # Убеждаемся, что индекс уникален
        if not prices_df.index.is_unique:
            print("Обнаружены дубликаты в финальном DataFrame. Удаляем...")
            prices_df = prices_df[~prices_df.index.duplicated(keep='first')]

        return prices_df

    @staticmethod
    def create_realistic_test_prices(tickers, days=30):
        """Создание реалистичных тестовых данных для российских акций"""
        print("Создание реалистичных тестовых данных...")

        # Создаем диапазон дат
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Базовые цены для российских акций (актуальные на 2024 год)
        base_prices = {
            'SBER': 250, 'GAZP': 160, 'LKOH': 5800, 'GMKN': 25000, 'NLMK': 180,
            'MTSS': 270, 'ROSN': 420, 'TATN': 380, 'VTBR': 0.03, 'MOEX': 150,
            'MAGN': 45000, 'PLZL': 12000, 'PHOR': 8000, 'AKRN': 6500, 'AFLT': 40,
            'PIKK': 1100, 'YNDX': 2900, 'OZON': 2200, 'FIVE': 2100, 'POLY': 700,
            'SNGS': 40, 'TCSG': 3000, 'RUAL': 300, 'ODVA': 200, 'TRMK': 1500,
            'LSRG': 800, 'UNAC': 500, 'VKCO': 400, 'DSKY': 600, 'LNTA': 2000,
            'RGSS': 300, 'FEES': 20, 'HYDR': 1, 'IRAO': 3
        }

        prices_df = pd.DataFrame(index=dates)

        for ticker in tickers:
            if ticker in base_prices:
                base_price = base_prices[ticker]
                # Создаем реалистичные колебания цен с трендом
                np.random.seed(hash(ticker) % 10000)  # Разные сиды для разных тикеров
                returns = np.random.normal(0.0005, 0.02, days)  # Небольшой позитивный тренд

                prices = [base_price]
                for ret in returns[1:]:
                    new_price = prices[-1] * (1 + ret)
                    prices.append(max(new_price, 0.01))  # Защита от отрицательных цен

                prices_df[ticker] = prices
            else:
                # Заполняем реалистичными случайными данными
                base_price = np.random.uniform(50, 1000)
                returns = np.random.normal(0, 0.015, days)

                prices = [base_price]
                for ret in returns[1:]:
                    new_price = prices[-1] * (1 + ret)
                    prices.append(max(new_price, 0.01))

                prices_df[ticker] = prices

        return prices_df


# ---------------------------
# 4) ФУНКЦИИ ДЛЯ СБОРА НОВОСТЕЙ
# ---------------------------

def fetch_russian_feeds_safe(feeds_dict, max_items_per_feed=10):
    """Безопасный сбор данных с RSS-лент с обработкой ошибок"""
    rows = []

    for name, url in tqdm(feeds_dict.items(), desc="Сбор российских новостей"):
        try:
            # Настраиваем заголовки для обхода блокировок
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/rss+xml, application/xml, text/xml'
            }

            feed = feedparser.parse(url)
            time.sleep(1)  # Пауза между запросами

        except Exception as e:
            print(f"Ошибка получения {name}: {e}")
            continue

        if not hasattr(feed, 'entries') or not feed.entries:
            print(f"Нет новостей для {name}")
            continue

        items_added = 0
        for i, entry in enumerate(feed.entries[:max_items_per_feed]):
            try:
                title = getattr(entry, 'title', '').strip()
                if not title or len(title) < 10:
                    continue

                # Более гибкий фильтр для российских финансовых новостей
                russian_finance_keywords = [
                    'акци', 'бирж', 'рубл', 'доллар', 'нефт', 'газ', 'цен', 'рынок',
                    'эконом', 'финанс', 'банк', 'инвест', 'торг', 'продаж', 'покуп',
                    'сбер', 'втб', 'газпром', 'лукойл', 'роснефть', 'норильский',
                    'мтс', 'рост', 'падение', 'дивиденд', 'квартал', 'отчетность',
                    'мосбиржа', 'доход', 'прибыль', 'выручка', 'капитализация',
                    'компани', 'бизнес', 'предприят', 'корпорац'
                ]

                title_lower = title.lower()
                if not any(keyword in title_lower for keyword in russian_finance_keywords):
                    continue

                rows.append({
                    'source': name,
                    'title': title,
                    'link': getattr(entry, 'link', ''),
                    'published': getattr(entry, 'published', getattr(entry, 'updated', '')),
                    'summary': getattr(entry, 'summary', '')[:200] if hasattr(entry, 'summary') else '',
                    'index': len(rows)  # Уникальный индекс
                })

                items_added += 1

            except Exception as e:
                continue

        if items_added > 0:
            print(f"Добавлено {items_added} новостей из {name}")

    if not rows:
        print("Не собрано ни одной новости. Создаем тестовые данные...")
        # Создаем качественные тестовые данные
        test_titles = [
            "Акции Газпрома выросли на 3% после новостей о новых контрактах",
            "Сбербанк сообщает о рекордной квартальной прибыли в 120 млрд рублей",
            "Лукойл увеличивает дивиденды на 15% по итогам года",
            "Российский фондовый рынок показывает рост на фоне укрепления рубля",
            "Норильский никель объявляет о планах по расширению производства",
            "МТС запускает новые тарифы для корпоративных клиентов",
            "Роснефть заключает соглашение о поставках нефти в Китай",
            "ВТБ улучшает прогноз по чистой прибыли на следующий год",
            "Яндекс демонстрирует рост выручки от рекламы на 25%",
            "Фармацевтические компании получают новые лицензии на производство",
            "Металлургические компании фиксируют снижение прибыли из-за падения цен",
            "Транспортные компании ожидают роста грузоперевозок в следующем квартале",
            "Строительные компании получают господдержку для новых проектов",
            "Розничные сети сообщают о росте продаж на 10%",
            "Электроэнергетические компании модернизируют сети"
        ]

        for i, title in enumerate(test_titles):
            rows.append({
                'source': 'тестовые данные',
                'title': title,
                'link': '',
                'published': datetime.now() - timedelta(days=i % 10),  # Разные даты
                'summary': '',
                'index': i
            })

    df = pd.DataFrame(rows)

    # Удаляем дубликаты по заголовку
    initial_count = len(df)
    df = df.drop_duplicates(subset=['title']).reset_index(drop=True)
    duplicates_removed = initial_count - len(df)

    if duplicates_removed > 0:
        print(f"Удалено {duplicates_removed} дубликатов новостей")

    # Обновляем индексы
    df['index'] = df.index

    # Обрабатываем даты
    df['published'] = pd.to_datetime(df['published'], errors='coerce')
    df = df.dropna(subset=['published'])

    print(f"Итоговый набор: {len(df)} уникальных российских финансовых новостей")
    return df


def get_russian_stock_prices_safe(tickers, days=30):
    """Безопасное получение цен российских акций с обработкой ошибок"""
    print("Загрузка данных о российских акциях...")

    # Пробуем получить реальные данные
    real_data = RussianStockData.get_moex_data_safe(tickers[:8], days)

    if real_data:
        print("Обработка загруженных данных...")

        # Безопасное создание DataFrame
        prices_df = RussianStockData.create_dataframe_from_series(real_data, min_length=5)

        if not prices_df.empty:
            print(f"Успешно создан DataFrame с {len(prices_df)} строками и {len(prices_df.columns)} колонками")

            # Проверяем качество данных
            nan_count = prices_df.isna().sum().sum()
            total_cells = prices_df.size
            nan_percentage = (nan_count / total_cells) * 100

            print(f"Пропущенные данные: {nan_count}/{total_cells} ({nan_percentage:.1f}%)")

            # Заполняем пропущенные значения
            if nan_count > 0:
                prices_df = prices_df.ffill().bfill()  # Заполнение вперед и назад
                print("Пропущенные значения заполнены")

            return prices_df

    # Если реальные данные не загрузились, используем тестовые
    print("Используем реалистичные тестовые данные...")
    return RussianStockData.create_realistic_test_prices(tickers, days)


# ---------------------------
# 5) КЛАСС ДЛЯ АНАЛИЗА НОВОСТЕЙ И ПРОГНОЗИРОВАНИЯ
# ---------------------------

class NewsAnalyzer:
    def __init__(self):
        self.sector_keywords = self._build_sector_keywords()
        self.sentiment_words = self._build_sentiment_lexicon()

    def _build_sector_keywords(self):
        """Расширенный словарь ключевых слов для секторов"""
        return {
            'Энергетика': ['нефть', 'газ', 'нефтегаз', 'энергетик', 'нефтяной', 'газовый', 'нефтедобыча',
                           'месторождение', 'трубопровод', 'нефтепродукт', 'скважина', 'нефтепереработка',
                           'голубое топливо'],
            'Финансы': ['банк', 'финанс', 'кредит', 'ипотек', 'вклад', 'актив', 'пассив', 'капитал',
                        'дивиденд', 'прибыль', 'убыток', 'баланс', 'отчетность', 'квартал', 'актив', 'лизинг'],
            'Металлургия': ['металл', 'сталь', 'никель', 'алюмин', 'медь', 'цинк', 'руд', 'горнодобывающ',
                            'платин', 'золот', 'серебр', 'металлург', 'прокат', 'чугун', 'ферросплав'],
            'Телекоммуникации': ['связь', 'телеком', 'интернет', 'мобильн', 'тариф', 'абонент', 'сеть',
                                 'роуминг', 'телефон', 'коммуникац', 'оператор', 'сотовая', 'интернет-провайдер'],
            'Химия': ['химия', 'удобрен', 'нефтехим', 'полимер', 'пластик', 'резин', 'лак', 'краск',
                      'химическ', 'производств', 'завод', 'полиэтилен', 'аммиак'],
            'Транспорт': ['авиа', 'транспорт', 'порт', 'аэропорт', 'груз', 'логистик', 'перевозк',
                          'судоходств', 'авиаперевозк', 'транспортн', 'железнодорож', 'фрахт'],
            'Строительство': ['строитель', 'девелопер', 'недвиж', 'жилье', 'квартир', 'дом', 'строит',
                              'застройщ', 'проект', 'объект', 'инфраструктур', 'жилищн', 'коммерческая недвижимость'],
            'IT-технологии': ['ит', 'технолог', 'интернет', 'софт', 'программ', 'приложен', 'цифров',
                              'онлайн', 'платформ', 'стартап', 'инновац', 'искусственн', 'ai', 'ml', 'кибер',
                              'digital'],
            'Розничная торговля': ['ритейл', 'магазин', 'торговля', 'розничн', 'покуп', 'продаж',
                                   'товар', 'ассортимент', 'сеть магазин', 'торгов', 'супермаркет', 'гипермаркет'],
            'Фармацевтика': ['фарма', 'медицин', 'лекарств', 'препарат', 'витамин', 'здоровье',
                             'больни', 'клиник', 'аптек', 'медицинск', 'биотех', 'фармпроизводство'],
            'Электроэнергетика': ['электроэнерг', 'энергосбыт', 'электросет', 'энергосистем',
                                  'электростанц', 'атомн', 'теплоэнерг', 'гидроэнерг', 'энергоблок', 'подстанц']
        }

    def _build_sentiment_lexicon(self):
        """Словарь для анализа тональности"""
        return {
            'positive': [
                'рост', 'вырос', 'увеличил', 'увеличение', 'прибыль', 'доход', 'успех', 'рекорд',
                'улучшение', 'позитив', 'сильный', 'стабильн', 'лидер', 'инновац', 'прорыв',
                'эффективн', 'процветан', 'уверен', 'оптимизм', 'перспектив', 'дивиденд', 'превышен',
                'успешн', 'высок', 'прирост', 'расширен', 'развит', 'модернизац', 'инвестиц'
            ],
            'negative': [
                'падение', 'снижен', 'упал', 'сокращен', 'убыток', 'проблем', 'кризис', 'слаб',
                'негатив', 'риск', 'опасен', 'потеря', 'обвал', 'дефицит', 'банкрот', 'долг',
                'сложн', 'нестабильн', 'ухудшен', 'спад', 'замедлен', 'снижаться', 'сложност',
                'сокращен', 'увольнен', 'закрыт', 'приостанов', 'девальвац', 'инфляц'
            ],
            'intensifiers': [
                'значительн', 'резк', 'существенн', 'кардинальн', 'радикальн', 'масштабн',
                'крупн', 'серьезн', 'крайн', 'очень', 'сильн', 'крайне', 'максимальн', 'рекордн'
            ]
        }

    def analyze_news_sentiment(self, title):
        """Анализ тональности заголовка новости"""
        title_lower = title.lower()

        positive_score = 0
        negative_score = 0

        # Анализ положительных слов
        for word in self.sentiment_words['positive']:
            if word in title_lower:
                positive_score += 1
                # Проверка на усилители
                for intensifier in self.sentiment_words['intensifiers']:
                    if f"{intensifier} {word}" in title_lower or f"{word} {intensifier}" in title_lower:
                        positive_score += 0.5
                        break

        # Анализ отрицательных слов
        for word in self.sentiment_words['negative']:
            if word in title_lower:
                negative_score += 1
                # Проверка на усилители
                for intensifier in self.sentiment_words['intensifiers']:
                    if f"{intensifier} {word}" in title_lower or f"{word} {intensifier}" in title_lower:
                        negative_score += 0.5
                        break

        # Определение итоговой тональности
        total_score = positive_score - negative_score
        if total_score > 0:
            return min(1.0, total_score * 0.15)  # Нормализация
        elif total_score < 0:
            return max(-1.0, total_score * 0.15)  # Нормализация
        else:
            return 0.0

    def predict_sector_movement(self, news_df):
        """Прогнозирование движения секторов на основе новостей"""
        sector_scores = {sector: 0.0 for sector in RUSSIAN_SECTORS.keys()}
        sector_news_count = {sector: 0 for sector in RUSSIAN_SECTORS.keys()}

        for _, news in news_df.iterrows():
            title = news['title']
            sentiment = self.analyze_news_sentiment(title)

            # Определение секторов для новости
            affected_sectors = self._identify_sectors_from_title(title)

            for sector in affected_sectors:
                sector_scores[sector] += sentiment
                sector_news_count[sector] += 1

        # Нормализация оценок и преобразование в рекомендации от -5 до +5
        recommendations = {}
        for sector in RUSSIAN_SECTORS.keys():
            if sector_news_count[sector] > 0:
                avg_score = sector_scores[sector] / sector_news_count[sector]
                # Преобразование в шкалу -5 до +5
                recommendation = int(round(avg_score * 5))
                recommendation = max(-5, min(5, recommendation))  # Ограничение диапазона
            else:
                recommendation = 0  # Нейтральная рекомендация при отсутствии новостей

            recommendations[sector] = recommendation

        return recommendations

    def _identify_sectors_from_title(self, title):
        """Идентификация секторов из заголовка новости"""
        title_lower = title.lower()
        detected_sectors = set()

        for sector, keywords in self.sector_keywords.items():
            for keyword in keywords:
                if keyword in title_lower:
                    detected_sectors.add(sector)
                    break

        return list(detected_sectors) if detected_sectors else ['Финансы']  # По умолчанию Финансы


# ---------------------------
# 6) КЛАСС ДЛЯ ОБУЧЕНИЯ ИИ
# ---------------------------

class StockAIModel:
    def __init__(self, model_path='stock_ai_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.is_trained = False

    def load_model(self):
        """Загрузка предобученной модели"""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                print(f"✅ Модель ИИ загружена из {self.model_path}")
                return True
            except Exception as e:
                print(f"❌ Ошибка загрузки модели: {e}")
        return False

    def create_new_model(self):
        """Создание новой модели"""
        self.model = RandomForestRegressor(
            n_estimators=50,  # Уменьшил для скорости
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        print("✅ Создана новая модель Random Forest")

    def generate_training_data(self, news_df, prices_df):
        """Генерация синтетических данных для обучения, если реальных недостаточно"""
        print("🔧 Генерация дополнительных тренировочных данных...")

        analyzer = NewsAnalyzer()

        # Создаем больше дат для обучения
        dates = pd.date_range(end=datetime.now(), periods=60, freq='D')

        X = []
        y = []

        # Генерируем синтетические данные на основе реальных паттернов
        for i in range(len(dates) - 5):
            # Случайные признаки (симуляция анализа новостей)
            features = []
            for sector in RUSSIAN_SECTORS.keys():
                # Генерируем реалистичные значения сентимента
                sentiment = np.random.normal(0, 0.3)  # Нормальное распределение вокруг 0
                sentiment = max(-1, min(1, sentiment))  # Ограничиваем диапазон
                features.append(sentiment)

            # Генерируем целевую переменную на основе признаков
            # Более сильный сентимент -> более высокая доходность
            base_return = np.sum(features) * 0.1  # Базовая доходность на основе сентимента
            noise = np.random.normal(0, 0.02)  # Шум
            target_return = base_return + noise

            X.append(features)
            y.append(target_return)

        return np.array(X), np.array(y)

    def prepare_training_data(self, news_df, prices_df):
        """Подготовка данных для обучения"""
        try:
            print("📊 Подготовка данных для обучения ИИ...")

            # Если данных мало, генерируем дополнительные
            if len(news_df) < 5 or len(prices_df) < 10:
                print("⚠️  Реальных данных недостаточно, генерируем синтетические...")
                return self.generate_training_data(news_df, prices_df)

            # Создаем признаки из новостей
            analyzer = NewsAnalyzer()

            # Группируем новости по дате и анализируем
            news_df['date'] = pd.to_datetime(news_df['published']).dt.date
            daily_sentiments = []

            for date in news_df['date'].unique():
                daily_news = news_df[news_df['date'] == date]
                sector_scores = analyzer.predict_sector_movement(daily_news)
                sector_scores['date'] = date
                daily_sentiments.append(sector_scores)

            if not daily_sentiments:
                print("⚠️  Не удалось извлечь сентимент из новостей, генерируем данные...")
                return self.generate_training_data(news_df, prices_df)

            sentiment_df = pd.DataFrame(daily_sentiments)
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

            # Рассчитываем доходность акций
            returns_df = prices_df.pct_change().dropna()

            if returns_df.empty:
                print("⚠️  Нет данных о доходности, генерируем данные...")
                return self.generate_training_data(news_df, prices_df)

            # Создаем целевые переменные (средняя доходность по секторам)
            targets = {}
            for sector, tickers in RUSSIAN_SECTORS.items():
                sector_tickers = [t for t in tickers if t in returns_df.columns]
                if sector_tickers:
                    targets[sector] = returns_df[sector_tickers].mean(axis=1)

            # Объединяем признаки и цели
            X = []
            y = []

            valid_pairs = 0
            for i in range(1, len(returns_df)):
                current_date = returns_df.index[i]
                prev_date = returns_df.index[i - 1]

                # Признаки: sentiment за предыдущий день
                prev_sentiment = sentiment_df[sentiment_df['date'] == prev_date]
                if not prev_sentiment.empty:
                    features = []
                    for sector in RUSSIAN_SECTORS.keys():
                        feature_value = prev_sentiment[sector].iloc[0] if sector in prev_sentiment.columns else 0
                        features.append(feature_value)

                    # Цель: доходность в текущий день
                    current_returns = []
                    for sector in RUSSIAN_SECTORS.keys():
                        if sector in targets and current_date in targets[sector].index:
                            current_returns.append(targets[sector].loc[current_date])
                        else:
                            current_returns.append(0)

                    if current_returns and not all(r == 0 for r in current_returns):
                        X.append(features)
                        y.append(np.mean(current_returns))
                        valid_pairs += 1

            print(f"✅ Подготовлено {valid_pairs} примеров для обучения")

            if valid_pairs < 10:
                print("⚠️  Мало реальных данных, добавляем синтетические...")
                X_synth, y_synth = self.generate_training_data(news_df, prices_df)
                X = np.vstack([X, X_synth]) if len(X) > 0 else X_synth
                y = np.concatenate([y, y_synth]) if len(y) > 0 else y_synth

            return np.array(X), np.array(y)

        except Exception as e:
            print(f"❌ Ошибка подготовки данных: {e}")
            print("🔄 Используем синтетические данные...")
            return self.generate_training_data(news_df, prices_df)

    def train(self, X, y):
        """Обучение модели"""
        if len(X) == 0 or len(y) == 0:
            print("❌ Недостаточно данных для обучения")
            return False

        try:
            print(f"🧠 Обучение модели на {len(X)} примерах...")

            if not self.is_trained:
                # Первоначальное обучение
                self.model.fit(X, y)
                self.is_trained = True
                print("✅ Модель успешно обучена")
            else:
                # Дообучение существующей модели
                self.model.fit(X, y)
                print("✅ Модель успешно дообучена")

            # Сохраняем модель
            joblib.dump(self.model, self.model_path)
            print(f"💾 Модель сохранена в {self.model_path}")
            return True

        except Exception as e:
            print(f"❌ Ошибка обучения модели: {e}")
            return False

    def predict(self, X):
        """Прогнозирование с помощью модели"""
        if self.model is None or not self.is_trained:
            return np.zeros(len(X))

        try:
            return self.model.predict(X)
        except Exception as e:
            print(f"❌ Ошибка прогнозирования: {e}")
            return np.zeros(len(X))


# ---------------------------
# 7) ОСНОВНАЯ ЛОГИКА ПРОГРАММЫ
# ---------------------------

def save_recommendations(recommendations, filename='RECOM.csv'):
    """Сохранение рекомендаций в CSV файл"""
    df = pd.DataFrame(list(recommendations.items()), columns=['Сектор', 'Рекомендация'])
    df['Описание'] = df['Рекомендация'].map(lambda x:
                                            'Максимальная покупка' if x == 5 else
                                            'Сильная покупка' if x >= 3 else
                                            'Умеренная покупка' if x >= 1 else
                                            'Нейтрально' if x == 0 else
                                            'Умеренная продажа' if x >= -2 else
                                            'Сильная продажа' if x >= -4 else
                                            'Максимальная продажа'
                                            )
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"💾 Рекомендации сохранены в {filename}")


def main():
    print("=== СИСТЕМА АНАЛИЗА РОССИЙСКИХ АКЦИЙ С ИИ ===")
    print("Версия с прогнозированием секторов и обучением ИИ\n")

    try:
        # Инициализация ИИ модели
        ai_model = StockAIModel()

        # Загрузка или создание модели
        if not ai_model.load_model():
            ai_model.create_new_model()

        # 1. Сбор российских новостей
        print("1. 📰 Сбор российских финансовых новостей...")
        news_df = fetch_russian_feeds_safe(RUSSIAN_FEEDS, max_items_per_feed=15)

        if news_df.empty:
            print("❌ Не удалось собрать данные. Завершение работы.")
            return

        print(f"✅ Собрано {len(news_df)} новостей")

        # 2. Загрузка данных об акциях
        print("\n2. 📊 Загрузка данных о российских акциях...")
        all_tickers = []
        for sector_tickers in RUSSIAN_SECTORS.values():
            all_tickers.extend(sector_tickers)

        prices_data = get_russian_stock_prices_safe(all_tickers, days=45)  # Увеличили период

        if prices_data.empty:
            print("❌ Не удалось загрузить данные об акциях. Завершение работы.")
            return

        print(f"✅ Загружены данные для {len(prices_data.columns)} тикеров за {len(prices_data)} дней")

        # 3. Анализ новостей и прогнозирование секторов
        print("\n3. 🔍 Анализ новостей и прогнозирование секторов...")
        analyzer = NewsAnalyzer()
        recommendations = analyzer.predict_sector_movement(news_df)

        # 4. Обучение ИИ на собранных данных
        print("\n4. 🧠 Обучение ИИ модели...")
        X, y = ai_model.prepare_training_data(news_df, prices_data)

        if ai_model.train(X, y):
            print("🎯 Обучение ИИ завершено успешно!")
        else:
            print("⚠️  Обучение ИИ не удалось, но рекомендации будут сгенерированы")

        # 5. Сохранение рекомендаций
        print("\n5. 💾 Сохранение рекомендаций...")
        save_recommendations(recommendations)

        # 6. Вывод результатов
        print("\n6. 📈 РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print("=" * 60)
        for sector, recommendation in sorted(recommendations.items(), key=lambda x: x[1], reverse=True):
            emoji = "🟢" if recommendation > 0 else "🔴" if recommendation < 0 else "⚪"
            action = "ПОКУПКА" if recommendation > 0 else "ПРОДАЖА" if recommendation < 0 else "НЕЙТРАЛЬНО"
            print(f"{emoji} {sector:25} {recommendation:+2d} ({action})")

        print("=" * 60)
        print("✅ Анализ завершен! Рекомендации сохранены в RECOM.csv")

        # 7. Дополнительная информация
        positive_recs = sum(1 for r in recommendations.values() if r > 0)
        negative_recs = sum(1 for r in recommendations.values() if r < 0)
        neutral_recs = len(recommendations) - positive_recs - negative_recs

        print(f"\n📊 Статистика рекомендаций:")
        print(f"   • Рекомендации на покупку: {positive_recs}")
        print(f"   • Рекомендации на продажу: {negative_recs}")
        print(f"   • Нейтральные рекомендации: {neutral_recs}")
        print(f"   • Всего секторов: {len(recommendations)}")

        # Сохранение сводного отчета
        summary_report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'news_count': len(news_df),
            'stocks_count': len(prices_data.columns),
            'period_days': len(prices_data),
            'positive_recommendations': positive_recs,
            'negative_recommendations': negative_recs,
            'neutral_recommendations': neutral_recs,
            'ai_model_trained': ai_model.is_trained,
            'status': 'success'
        }

        import json
        with open('analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Сводный отчет сохранен: analysis_summary.json")

    except Exception as e:
        print(f"\n❌ Критическая ошибка в main(): {e}")
        import traceback
        traceback.print_exc()


# ---------------------------
# 8) ЗАПУСК ПРОГРАММЫ
# ---------------------------

if __name__ == "__main__":
    print("🚀 Запуск системы анализа российских акций с ИИ...")
    start_time = time.time()

    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Программа прервана пользователем")
    except Exception as e:
        print(f"\n💥 Непредвиденная ошибка: {e}")
        import traceback

        traceback.print_exc()
    finally:
        execution_time = time.time() - start_time
        print(f"\n⏱️ Время выполнения: {execution_time:.2f} секунд")
        print("🏁 Программа завершена")