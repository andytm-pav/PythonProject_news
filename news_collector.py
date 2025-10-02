import time
from datetime import datetime
import feedparser
import pandas as pd

import library as l


# ==================== КЛАСС ДЛЯ СБОРА НОВОСТЕЙ ====================
class NewsCollector:
    def __init__(self):
        self.feeds = l.RUSSIAN_FEEDS

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
