# Freedom Hackathon 2026 — Churn Intelligence

## Быстрый старт

### 1. Установить зависимости
```
pip install -r requirements.txt
```

### 2. Обучить модель (занимает ~2-3 минуты на CPU, ~30 сек на GPU)
```
set PYTHONIOENCODING=utf-8
python churn_pipeline.py
```

### 3. Запустить dashboard
```
streamlit run dashboard.py
```

---

## Результаты модели

| Метрика | Значение |
|---------|---------|
| AUC-ROC | **0.8086** |
| F1-score | 0.6538 |
| Precision | 0.5555 |
| Recall | 0.7942 |

## Сегментация (Uplift)

| Сегмент | Пользователей | Avg P(churn) | Total pLTV |
|---------|--------------|-------------|------------|
| Sure Things | 89,569 | 17% | 5.66B |
| **Persuadables** | **97,065** | **63%** | **322M** |
| Lost Causes | 125,771 | 67% | 0.28M |
| Low Value Stable | 60,875 | 20% | 0.1M |
| Sleeping Dogs | 24 | 66% | — |

## Топ-факторы оттока (SHAP)
1. `proxy_ltv_90d` — немонетизированные пользователи уходят чаще
2. `evt_trend` — снижение активности в приложении
3. `age` — возраст влияет на паттерн удержания
4. `transfer_evt_7d` — P2P переводы = вовлечённость
5. `open_card_7d` — открытие карты = сигнал лояльности

## Структура файлов
```
churn_pipeline.py    — обучение модели
dashboard.py         — Streamlit dashboard
outputs/
  catboost_churn.cbm         — обученная модель
  users_scored.csv           — все пользователи с P(churn) и сегментом
  shap_bar.png               — feature importance
  shap_beeswarm.png          — направление влияния факторов
  uplift_segments_summary.csv
  retention_strategies.csv
  metrics.csv
```
