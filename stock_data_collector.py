import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests


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
