import threading
import time
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import library as l
import news_analyzer as na
import news_collector as nc
import stock_data_collector as sdc
from enhanced_stock_ai_model import EnhancedStockAIModel
from portfolio_rebalancer import PortfolioRebalancer
from virtual_portfolio import VirtualPortfolio
from web_interface import WebInterface

warnings.filterwarnings('ignore')


class EnhancedStockAnalysisSystem:
    def __init__(self):
        self.news_collector = nc.NewsCollector()
        self.stock_collector = sdc.StockDataCollector()
        self.news_analyzer = na.NewsAnalyzer()
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
        """Улучшенный цикл анализа"""
        print("\n" + "=" * 60)
        print(f"🔍 Запуск анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        try:
            # Выводим статус модели перед началом
            model_status = self.enhanced_ai_model.get_model_status()
            print(f"🤖 Статус модели: {model_status}")

            # 1. Сбор данных
            print("📰 Шаг 1: Сбор новостей...")
            news_df = self.news_collector.fetch_all_news()

            print("📊 Шаг 2: Сбор данных об акциях...")
            all_tickers = [ticker for sector_tickers in l.RUSSIAN_SECTORS.values() for ticker in sector_tickers]
            prices_df = self.stock_collector.get_stock_prices(all_tickers)

            # 2. Анализ новостей
            print("🔍 Шаг 3: Анализ новостей...")
            news_recommendations = self.news_analyzer.predict_sector_sentiment(news_df)

            # 3. Обучение AI моделей с дообучением
            print("🎯 Шаг 4: Обучение AI моделей...")
            news_sentiment_by_date = {datetime.now().date(): news_recommendations}

            # Определяем тип обучения: дообучение если модель уже обучена
            incremental_learning = self.enhanced_ai_model.is_trained
            training_successful = self.enhanced_ai_model.train_models(
                prices_df,
                news_sentiment_by_date,
                incremental=incremental_learning
            )

            if not training_successful:
                print("⚠️ Обучение не удалось, используем базовые рекомендации")
            else:
                training_type = "дообучение" if incremental_learning else "первоначальное обучение"
                print(f"✅ Модели успешно прошли {training_type}")

            # 4. AI анализ
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

            # 5. Ребалансировка
            print("⚖️ Шаг 6: Проверка ребалансировки портфеля...")
            rebalancing_trades = self.portfolio_rebalancer.execute_rebalancing(current_prices, news_recommendations)

            # 6. Рекомендации
            enhanced_recommendations = self._enhance_recommendations_with_ai(news_recommendations, ai_predictions)

            # 7. Сохранение и отображение
            portfolio_value = self.portfolio.get_portfolio_value(current_prices)
            self.save_backup_data(news_df, prices_df, enhanced_recommendations)
            self.web_interface.update_data(enhanced_recommendations, news_df, portfolio_value)

            # 8. Вывод результатов
            self._print_enhanced_results(enhanced_recommendations, ai_predictions, rebalancing_trades, portfolio_value)

            return True

        except Exception as e:
            print(f"💥 Ошибка в цикле анализа: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def _enhance_recommendations_with_ai(news_recommendations, ai_predictions):
        """Улучшение рекомендаций с помощью AI прогнозов"""
        enhanced_recommendations = news_recommendations.copy()

        sector_predictions = {}
        for ticker_data in ai_predictions.values():
            sector = ticker_data['sector']
            if sector not in sector_predictions:
                sector_predictions[sector] = []

            weighted_prediction = ticker_data['predicted_return'] * ticker_data['confidence']
            sector_predictions[sector].append(weighted_prediction)

        for sector, predictions in sector_predictions.items():
            if predictions:
                avg_prediction = np.mean(predictions)
                ai_adjustment = int(round(avg_prediction * 200))
                ai_adjustment = max(-5, min(5, ai_adjustment))

                if sector in enhanced_recommendations:
                    enhanced_recommendations[sector] = int(
                        0.3 * enhanced_recommendations[sector] + 0.7 * ai_adjustment
                    )
                else:
                    enhanced_recommendations[sector] = ai_adjustment

        return enhanced_recommendations

    def _print_enhanced_results(self, recommendations, ai_predictions, trades, portfolio_value):
        """Вывод результатов анализа"""
        print("\n📈 РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print("-" * 60)

        print("🎯 РЕКОМЕНДАЦИИ ПО СЕКТОРАМ:")
        for sector, rec in sorted(recommendations.items(), key=lambda x: x[1], reverse=True):
            emoji = "🟢" if rec > 0 else "🔴" if rec < 0 else "⚪"
            print(f"{emoji} {sector:30} {rec:+2d}/5")

        if trades:
            print(f"\n⚖️ ВЫПОЛНЕННЫЕ СДЕЛКИ: {len(trades)}")

        formatted_value = f"₽{portfolio_value:,.2f}"
        print(f"\n💼 ТЕКУЩАЯ СТОИМОСТЬ ПОРТФЕЛЯ: {formatted_value}")

        # Выводим информацию о модели
        model_info = self.enhanced_ai_model.get_training_info()
        print(f"🤖 ИНФОРМАЦИЯ О МОДЕЛИ: {model_info['total_trainings']} тренировок, "
              f"{model_info['samples']} samples")

        print("✅ АНАЛИЗ ЗАВЕРШЕН!")

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

    def run_continuous_analysis(self, interval_minutes=30):
        """Запуск непрерывного анализа"""
        print("🚀 Запуск системы анализа российских акций с ИИ...")

        web_thread = threading.Thread(target=self.web_interface.run_server)
        web_thread.daemon = True
        web_thread.start()

        cycle_count = 0
        while self.web_interface.is_running:
            try:
                success = self.run_enhanced_analysis_cycle()
                cycle_count += 1

                if success:
                    print(f"\n♻️ Цикл {cycle_count} завершен. Ожидание...")
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
                print(f"💥 Ошибка: {e}")
                time.sleep(300)

        print("🛑 Программа остановлена")


# ==================== ЗАПУСК ПРОГРАММЫ ====================
if __name__ == "__main__":
    system = EnhancedStockAnalysisSystem()
    system.run_continuous_analysis(interval_minutes=1)
