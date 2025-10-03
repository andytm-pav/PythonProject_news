from datetime import datetime
import json
from flask import Flask, render_template_string, request, jsonify

# ==================== ВЕБ-ИНТЕРФЕЙС ====================
app = Flask(__name__)


class WebInterface:
    def __init__(self):
        self.current_recommendations = {}
        self.news_items = []
        self.portfolio_value = 0
        self.is_running = True
        # Данные для графиков
        self.stock_data = {}  # {sector: {'predicted': [], 'actual': [], 'iterations': []}}
        self.portfolio_data = {'predicted': [], 'actual': [], 'iterations': []}
        self.iteration_count = 0

    def update_data(self, recommendations, news_df, portfolio_value, stock_predictions=None, actual_prices=None):
        """Обновление данных для отображения"""
        self.current_recommendations = recommendations
        self.news_items = news_df.to_dict('records') if not news_df.empty else []
        self.portfolio_value = portfolio_value

        # Обновление данных для графиков
        self.iteration_count += 1

        # Обновление данных по акциям
        if stock_predictions and actual_prices:
            for sector, predicted_price in stock_predictions.items():
                if sector not in self.stock_data:
                    self.stock_data[sector] = {'predicted': [], 'actual': [], 'iterations': []}

                actual_price = actual_prices.get(sector, 0)
                self.stock_data[sector]['predicted'].append(float(predicted_price))
                self.stock_data[sector]['actual'].append(float(actual_price))
                self.stock_data[sector]['iterations'].append(self.iteration_count)

        # Обновление данных портфеля (упрощенная логика)
        self.portfolio_data['predicted'].append(float(portfolio_value * 1.1))  # пример предсказанной стоимости
        self.portfolio_data['actual'].append(float(portfolio_value))
        self.portfolio_data['iterations'].append(self.iteration_count)

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
                .sector-card { background: white; padding: 15px; border-radius: 8px; border-left: 5px solid #ddd; cursor: pointer; transition: transform 0.2s; }
                .sector-card:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                .positive { border-left-color: #4CAF50 !important; background: #f1f8e9 !important; }
                .negative { border-left-color: #f44336 !important; background: #ffebee !important; }
                .news-item { background: white; padding: 15px; margin-bottom: 10px; border-radius: 8px; border-left: 4px solid #2196F3; }
                .news-new { background: #e3f2fd !important; border-left-color: #FF9800 !important; }
                .controls { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .btn { padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 5px; cursor: pointer; margin-right: 10px; }
                .btn-stop { background: #f44336; }
                .btn-graph { background: #9C27B0; }
                .portfolio { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .graph-buttons { display: flex; gap: 10px; margin-top: 10px; }
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
                    <button class="btn btn-graph" onclick="window.open('/portfolio-graph', '_blank')">График портфеля</button>
                    <button class="btn btn-stop" onclick="stopProgram()">Остановить программу</button>
                </div>

                <div class="portfolio">
                    <h2>💼 Виртуальный портфель</h2>
                    <p><strong>Текущая стоимость:</strong> {{ "₽{:,.2f}".format(portfolio_value) }}</p>
                </div>

                <h2>📈 Рекомендации по секторам</h2>
                <div class="recommendations">
                    {% for sector, rec in recommendations.items() %}
                    <div class="sector-card {{ 'positive' if rec > 0 else 'negative' if rec < 0 else '' }}" 
                         onclick="showSectorGraph('{{ sector }}')">
                        <h3>{{ sector }}</h3>
                        <p><strong>Рекомендация:</strong> {{ rec }}/5</p>
                        <p><em>{{ get_recommendation_text(rec) }}</em></p>
                        <div class="graph-buttons">
                            <button class="btn" onclick="event.stopPropagation(); showSectorGraph('{{ sector }}')">
                                Показать график
                            </button>
                        </div>
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

                function showSectorGraph(sector) {
                    window.open('/sector-graph?name=' + encodeURIComponent(sector), '_blank');
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

        @app.route('/sector-graph')
        def sector_graph():
            """Страница с графиком для конкретного сектора"""
            sector_name = request.args.get('name', '')
            return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>График акций - {{ sector_name }}</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                    .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
                    .btn { padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <button class="btn" onclick="window.close()">Закрыть</button>
                    <h1>График стоимости акций: {{ sector_name }}</h1>
                    <canvas id="stockChart" width="800" height="400"></canvas>
                </div>

                <script>
                    fetch('/sector-data?name={{ sector_name }}')
                        .then(response => response.json())
                        .then(data => {
                            const ctx = document.getElementById('stockChart').getContext('2d');
                            new Chart(ctx, {
                                type: 'line',
                                data: {
                                    labels: data.iterations,
                                    datasets: [
                                        {
                                            label: 'Предполагаемая стоимость (руб)',
                                            data: data.predicted,
                                            borderColor: '#4CAF50',
                                            backgroundColor: 'rgba(76, 175, 80, 0.1)',
                                            borderWidth: 2,
                                            fill: true
                                        },
                                        {
                                            label: 'Фактическая стоимость (руб)',
                                            data: data.actual,
                                            borderColor: '#f44336',
                                            backgroundColor: 'rgba(244, 67, 54, 0.1)',
                                            borderWidth: 2,
                                            fill: true
                                        }
                                    ]
                                },
                                options: {
                                    responsive: true,
                                    scales: {
                                        y: {
                                            beginAtZero: false,
                                            title: {
                                                display: true,
                                                text: 'Стоимость (рубли)'
                                            }
                                        },
                                        x: {
                                            title: {
                                                display: true,
                                                text: 'Итерации'
                                            }
                                        }
                                    }
                                }
                            });
                        });
                </script>
            </body>
            </html>
            ''', sector_name=sector_name)

        @app.route('/portfolio-graph')
        def portfolio_graph():
            """Страница с графиком портфеля"""
            return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>График стоимости портфеля</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                    .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
                    .btn { padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <button class="btn" onclick="window.close()">Закрыть</button>
                    <h1>График стоимости портфеля</h1>
                    <canvas id="portfolioChart" width="800" height="400"></canvas>
                </div>

                <script>
                    fetch('/portfolio-data')
                        .then(response => response.json())
                        .then(data => {
                            const ctx = document.getElementById('portfolioChart').getContext('2d');
                            new Chart(ctx, {
                                type: 'line',
                                data: {
                                    labels: data.iterations,
                                    datasets: [
                                        {
                                            label: 'Предполагаемая стоимость портфеля (руб)',
                                            data: data.predicted,
                                            borderColor: '#9C27B0',
                                            backgroundColor: 'rgba(156, 39, 176, 0.1)',
                                            borderWidth: 2,
                                            fill: true
                                        },
                                        {
                                            label: 'Фактическая стоимость портфеля (руб)',
                                            data: data.actual,
                                            borderColor: '#FF9800',
                                            backgroundColor: 'rgba(255, 152, 0, 0.1)',
                                            borderWidth: 2,
                                            fill: true
                                        }
                                    ]
                                },
                                options: {
                                    responsive: true,
                                    scales: {
                                        y: {
                                            beginAtZero: false,
                                            title: {
                                                display: true,
                                                text: 'Стоимость (рубли)'
                                            }
                                        },
                                        x: {
                                            title: {
                                                display: true,
                                                text: 'Итерации'
                                            }
                                        }
                                    }
                                }
                            });
                        });
                </script>
            </body>
            </html>
            ''')

        @app.route('/sector-data')
        def sector_data():
            """API для получения данных по сектору"""
            sector_name = request.args.get('name', '')
            data = self.stock_data.get(sector_name, {'predicted': [], 'actual': [], 'iterations': []})
            return jsonify(data)

        @app.route('/portfolio-data')
        def portfolio_data():
            """API для получения данных портфеля"""
            return jsonify(self.portfolio_data)

        @app.route('/stop', methods=['POST'])
        def stop_program():
            self.is_running = False
            return {'success': True}

        print(f"🚀 Веб-интерфейс доступен по адресу: http://{host}:{port}")
        app.run(host=host, port=port, debug=False, use_reloader=False)