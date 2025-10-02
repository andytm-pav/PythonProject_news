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

# URL претренированных моделей (пример)
PRETRAINED_MODEL_URLS = {
    'base_model': 'https://example.com/models/russian_stocks_base_model.joblib',
    'advanced_model': 'https://example.com/models/russian_stocks_advanced_model.joblib'
}
