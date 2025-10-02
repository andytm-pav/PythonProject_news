from datetime import datetime

import library as l


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
        sectors_count = len(l.RUSSIAN_SECTORS)

        for sector in l.RUSSIAN_SECTORS.keys():
            sector_weights[sector] = 1.0 / sectors_count

        return sector_weights

    def calculate_current_weights(self, current_prices):
        """Расчет текущих весов портфеля"""
        total_value = self.portfolio.get_portfolio_value(current_prices)
        if total_value == 0:
            return {}

        current_weights = {}
        sector_values = {sector: 0 for sector in l.RUSSIAN_SECTORS.keys()}

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
                sector_tickers = l.RUSSIAN_SECTORS.get(sector, [])

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
