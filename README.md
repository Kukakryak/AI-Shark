# AI Shark

Дипломный проект: система интеллектуального анализа сетевого трафика с использованием нейронных сетей для выявления атак на основе CIC-IDS 2018.

## Что сделано в коде

- убрана утечка данных: `MinMaxScaler` обучается только на train-части;
- train/test делятся по порядку строк, без случайного перемешивания пересекающихся LSTM-окон;
- подготовка данных, обучение, оценка и сохранение артефактов вынесены в один pipeline;
- артефакты обучения сохраняются в каталог `artifacts/`:
  - `lstm_ids_model.keras`
  - `scaler.pkl`
  - `metadata.json`
  - `training_history.png`

## Запуск

Рекомендуется Python 3.11 и отдельное виртуальное окружение.

```bash
python preprocessing.py --dataset datasets/02-14-2018.csv --epochs 30
```

Более честная схема для диплома: обучать на одних днях, тестировать на других.

```bash
python preprocessing.py ^
  --train-datasets datasets/02-14-2018.csv datasets/02-15-2018.csv ^
  --test-datasets datasets/02-16-2018.csv ^
  --epochs 30
```

Для быстрого smoke-run на части датасета:

```bash
python preprocessing.py --dataset datasets/02-14-2018.csv --epochs 3 --max-rows 50000
```

Локальные unit-тесты:

```bash
python test.py
```

## Эксперименты для диплома

Построить профиль классов по всем CSV:

```bash
python experiment.py --profile-only
```

Запустить набор коротких сценариев и получить сводную таблицу:

```bash
python experiment.py --epochs 1
```

В сводку теперь автоматически попадают сравнения:

- `lstm`
- `logreg`
- `random_forest`

Для быстрого прогона сценариев на урезанных подвыборках:

```bash
python experiment.py --epochs 1 --fast-max-rows 50000
```

Сводные файлы сохраняются в `reports/`:

- `dataset_profile.csv` - распределение классов по каждому CSV;
- `experiment_summary.csv` - итоги сценариев;
- `reports/<scenario>/metadata.json` и модельные артефакты для каждого сценария.

## Ограничения и идеи для дипломной доработки

- сравнить LSTM с baseline-моделями на табличных данных: Logistic Regression, Random Forest, XGBoost;
- сохранить отдельный inference-скрипт для предсказаний на новых CSV;
- добавить анализ дисбаланса классов и сравнение macro/micro/weighted метрик;
- описать в тексте диплома, почему выбран временной split, а не случайный.
