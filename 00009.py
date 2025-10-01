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

# ==================== КОНФИГУРАЦИЯ RSS-ЛЕНТ ====================
RUSSIAN_FEEDS = {
    "Ведомости": "https://www.vedomosti.ru/rss/news",
    "Коммерсант": "https://www.kommersant.ru/RSS/news.xml",
    "Интерфакс": "https://www.interfax.ru/rss.asp",
    "ТАСС": "https://tass.ru/rss/v2.xml",
    "Лента.ру": "https://lenta.ru/rss/news",
    "Финам": "https://www.finam.ru/analysis/conews/rsspoint/",
    "БФМ": "https://www.bfm.ru/news.rss",
    "Инвестирование": "https://ru.investing.com/rss/news_25.rss"
}

# ==================== КОНФИГУРАЦИЯ СЕКТОРОВ И АКЦИЙ ====================
RUSSIAN_SECTORS = {
    'Энергетика': ['GAZP', 'LKOH', 'ROSN', 'SNGS', 'TATN'],
    'Индустриальный': ['FLOT', 'NMTP', 'TRMK', 'KMAZ'],
    'Базовые ресурсы': ['GMKN', 'NLMK', 'MAGN', 'PLZL', 'RUAL'],
    'Розничная торговля': ['MGNT', 'FIVE', 'DSKY', 'LNTA'],
    'Медицина': ['PHOR', 'POLY', 'RGSS'],
    'Потребительский': ['BELU', 'WTCM', 'UPRO'],
    'Телекоммуникации': ['MTSS', 'RTKM', 'MFON'],
    'Химия': ['HIMC', 'KAZT'],
    'Электроэнергетика': ['FEES', 'HYDR', 'IRAO', 'TGKA'],
    'Строительство': ['PIKK', 'LSRG', 'UNAC'],
    'Транспорт': ['AFLT', 'NKHP']
}


# ==================== ФУНКЦИЯ СБОРА НОВОСТЕЙ С ПРОВЕРКАМИ ====================
def fetch_russian_feeds_safe(feeds_dict, max_items_per_feed=10):
    """Безопасный сбор данных с RSS-лент с обработкой ошибок"""
    if not feeds_dict:
        print("❌ Словарь RSS-лент пуст или не определен")
        return pd.DataFrame()

    rows = []

    for name, url in tqdm(feeds_dict.items(), desc="Сбор новостей"):
        try:
            feed = feedparser.parse(url)
            time.sleep(1)

            if not hasattr(feed, 'entries') or not feed.entries:
                continue

            for entry in feed.entries[:max_items_per_feed]:
                title = getattr(entry, 'title', '').strip()
                if title:
                    rows.append({
                        'source': name,
                        'title': title,
                        'link': getattr(entry, 'link', ''),
                        'published': getattr(entry, 'published', ''),
                        'summary': getattr(entry, 'summary', '')[:200]
                    })

        except Exception as e:
            print(f"Ошибка при обработке {name}: {e}")
            continue

    if not rows:
        print("⚠️ Не собрано ни одной новости, создаю тестовые данные")
        return create_sample_news_data()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=['title'])
    df['published'] = pd.to_datetime(df['published'], errors='coerce')
    df = df.dropna(subset=['published'])

    print(f"✅ Собрано {len(df)} новостей")
    return df


def create_sample_news_data():
    """Создание тестовых новостей если реальные недоступны"""
    sample_titles = [
        "Акции Газпрома растут на фоне роста цен на нефть",
        "Сбербанк увеличивает дивиденды по итогам года",
        "Российский рынок акций показывает уверенный рост",
        "Компании металлургического сектора увеличивают производство",
        "Инвесторы проявляют интерес к акциям IT-сектора"
    ]

    data = []
    for i, title in enumerate(sample_titles):
        data.append({
            'source': 'тест',
            'title': title,
            'link': '',
            'published': datetime.now() - timedelta(hours=i),
            'summary': '',
            'index': i
        })

    return pd.DataFrame(data)


# ==================== ФУНКЦИЯ ПОЛУЧЕНИЯ ЦЕН АКЦИЙ С ПРОВЕРКАМИ ====================
def get_russian_stock_prices_safe(tickers, days=30):
    """Безопасное получение цен акций"""
    try:
        # Здесь должна быть ваша реализация получения данных с MOEX
        # Сейчас создадим реалистичные тестовые данные

        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        prices_df = pd.DataFrame(index=dates)

        # Базовые цены для российских акций
        base_prices = {
            'SBER': 250, 'GAZP': 160, 'LKOH': 5800, 'GMKN': 25000,
            'NLMK': 180, 'MTSS': 270, 'ROSN': 420, 'TATN': 380
        }

        for ticker in tickers:
            if ticker in base_prices:
                base_price = base_prices[ticker]
                returns = np.random.normal(0.001, 0.02, days)
                prices = [base_price]
                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))
                prices_df[ticker] = prices

        print(f"✅ Созданы тестовые данные для {len(prices_df.columns)} тикеров")
        return prices_df

    except Exception as e:
        print(f"❌ Ошибка создания тестовых данных: {e}")
        return None


# ==================== ОСНОВНОЙ КЛАСС АНАЛИЗАТОРА ====================
class NewsAnalyzer:
    def __init__(self):
        self.sector_keywords = self._build_sector_keywords()

    def _build_sector_keywords(self):
        return {
            'Энергетика': ['нефть', 'газ', 'нефтегаз', 'энергетик'],
            'Индустриальный': ['финанс', 'банк', 'инвест', 'кредит'],
            # ... добавьте ключевые слова для всех секторов
        }

    def predict_sector_movement(self, news_df):
        """Упрощенный прогноз движения секторов"""
        recommendations = {}
        for sector in RUSSIAN_SECTORS.keys():
            # Базовая логика анализа - замените на вашу
            recommendations[sector] = np.random.randint(-2, 3)
        return recommendations


# ==================== ОСНОВНАЯ ФУНКЦИЯ С ОБРАБОТКОЙ ОШИБОК ====================
def main():
    print("=== СИСТЕМА АНАЛИЗА РОССИЙСКОГО РЫНКА АКЦИЙ ===")

    analyzer = NewsAnalyzer()
    iteration = 0

    while True:
        try:
            print(f"\n🔄 Итерация анализа #{iteration + 1}")

            # 1. Сбор новостей с проверкой
            print("📰 Сбор новостей...")
            news_df = fetch_russian_feeds_safe(RUSSIAN_FEEDS)

            if news_df.empty:
                print("⚠️ Новости не собраны, пропускаю итерацию")
                time.sleep(60)
                continue

            # 2. Получение данных об акциях с проверкой
            print("📊 Загрузка данных об акциях...")
            all_tickers = []
            for sector_tickers in RUSSIAN_SECTORS.values():
                all_tickers.extend(sector_tickers)

            prices_data = get_russian_stock_prices_safe(all_tickers)

            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: проверка на None перед использованием
            if prices_data is None:
                print("❌ Данные об акциях не загружены, пропускаю итерацию")
                time.sleep(60)
                continue

            # 3. Проверка перед использованием iterrows
            if not hasattr(prices_data, 'iterrows'):
                print("❌ Объект prices_data не поддерживает iterrows")
                time.sleep(60)
                continue

            # 4. Безопасное использование iterrows
            try:
                for index, row in prices_data.iterrows():
                    # Ваша логика обработки данных
                    pass
            except Exception as e:
                print(f"❌ Ошибка в iterrows: {e}")

            # 5. Анализ и рекомендации
            print("🔍 Анализ новостей...")
            recommendations = analyzer.predict_sector_movement(news_df)

            # 6. Сохранение результатов
            save_recommendations(recommendations)

            print("✅ Итерация завершена успешно")
            iteration += 1
            time.sleep(300)  # Пауза 5 минут

        except KeyboardInterrupt:
            print("\n⚠️ Программа остановлена пользователем")
            break
        except Exception as e:
            print(f"❌ Непредвиденная ошибка: {e}")
            time.sleep(60)


def save_recommendations(recommendations, filename='RECOM.csv'):
    """Сохранение рекомендаций в CSV"""
    try:
        df = pd.DataFrame(list(recommendations.items()),
                          columns=['Сектор', 'Рекомендация'])
        df['Временная_метка'] = datetime.now()
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"💾 Рекомендации сохранены в {filename}")
    except Exception as e:
        print(f"❌ Ошибка сохранения рекомендаций: {e}")


if __name__ == "__main__":
    main()