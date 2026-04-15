# AI Shark

Дипломный проект: система интеллектуального анализа сетевого трафика с использованием нейронных сетей для выявления сетевых атак на основе датасета CIC-IDS 2018.

## О проекте

Проект решает задачу классификации сетевого трафика:

- вход: CSV-файлы с признаками сетевых соединений;
- выход: предсказание класса трафика, например `Benign`, `Bot`, `FTP-BruteForce`, `DoS attacks-SlowHTTPTest`;
- основная модель: `LSTM`, работающая не с одной строкой, а с последовательностью из нескольких подряд идущих записей;
- дополнительные модели для сравнения: `Logistic Regression` и `Random Forest`.

Идея проекта в том, что сетевой трафик можно рассматривать как поток событий во времени. Поэтому основная модель получает на вход окно длиной `sequence_length` и предсказывает класс последней записи в этом окне.

## Что лежит в репозитории

- [preprocessing.py](/C:/Users/Kukakryak/PycharmProjects/ai_shark/preprocessing.py:1) - основной pipeline: загрузка данных, очистка, кодирование классов, нормализация, построение последовательностей, обучение LSTM, оценка и сохранение артефактов.
- [experiment.py](/C:/Users/Kukakryak/PycharmProjects/ai_shark/experiment.py:1) - запуск экспериментальных сценариев и сравнение `lstm`, `logreg`, `random_forest`.
- [test.py](/C:/Users/Kukakryak/PycharmProjects/ai_shark/test.py:1) - unit-тесты на ключевые этапы подготовки данных.
- [PROJECT_GUIDE.md](/C:/Users/Kukakryak/PycharmProjects/ai_shark/PROJECT_GUIDE.md:1) - подробный учебный разбор проекта.
- [DIPLOMA_RESULTS.md](/C:/Users/Kukakryak/PycharmProjects/ai_shark/DIPLOMA_RESULTS.md:1) - заготовка выводов и результатов для диплома.
- [FINAL_EXECUTION_PLAN.md](/C:/Users/Kukakryak/PycharmProjects/ai_shark/FINAL_EXECUTION_PLAN.md:1) - план финальных запусков и оформления.
- `datasets/` - CSV-файлы CIC-IDS 2018.

## Как работает pipeline

Общий процесс такой:

1. Загружается один CSV или несколько CSV.
2. Определяется столбец с метками классов, обычно это `Label`.
3. Данные очищаются:
   - удаляются мусорные строки;
   - убираются `NaN`, `inf`, дубликаты заголовка внутри таблицы;
   - отбрасываются некорректные записи.
4. Метки классов кодируются в числа.
5. Данные делятся на `train/test` без случайного перемешивания.
6. `MinMaxScaler` обучается только на train-части.
7. Из строк строятся последовательности фиксированной длины для LSTM.
8. Модель обучается и оценивается по метрикам.
9. Сохраняются артефакты эксперимента.

Это сделано специально, чтобы избежать утечки данных между train и test.

## Почему выбран LSTM

В отличие от обычной табличной классификации, LSTM умеет учитывать порядок событий. Для задачи анализа сетевого трафика это полезно, потому что атаки часто проявляются не одной строкой, а характерной последовательностью соединений.

При этом в проекте есть и baseline-модели. Это важно для диплома: можно показать не только работу нейросети, но и сравнение с более простыми методами.

## Требования

- Python `3.11`
- Windows PowerShell или любой терминал с Python
- установленное виртуальное окружение `.venv`

Зависимости описаны в [requirements.txt](/C:/Users/Kukakryak/PycharmProjects/ai_shark/requirements.txt:1).

## Установка с нуля

### 1. Создать виртуальное окружение

```powershell
py -3.11 -m venv .venv
```

### 2. Активировать его

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Установить зависимости

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Подготовить датасеты

Положи CSV-файлы CIC-IDS 2018 в каталог `datasets/`.

Примеры файлов, с которыми уже работает проект:

- `02-14-2018.csv`
- `02-15-2018.csv`
- `02-16-2018.csv`
- `02-22-2018.csv`
- `02-23-2018.csv`
- `02-28-2018.csv`
- `03-01-2018.csv`
- `03-02-2018.csv`

## Быстрый старт

### Проверка тестов

```powershell
.\.venv\Scripts\python.exe test.py
```

### Быстрый smoke-run обучения

```powershell
.\.venv\Scripts\python.exe preprocessing.py --dataset datasets/03-02-2018.csv --epochs 1 --max-rows 50000 --output-dir artifacts_smoke
```

### Базовый запуск обучения

```powershell
.\.venv\Scripts\python.exe preprocessing.py --dataset datasets/03-02-2018.csv --epochs 10 --output-dir artifacts_bot
```

## Режимы запуска обучения

### 1. Обучение на одном CSV

```powershell
.\.venv\Scripts\python.exe preprocessing.py --dataset datasets/02-14-2018.csv --epochs 10
```

### 2. Обучение на нескольких CSV

```powershell
.\.venv\Scripts\python.exe preprocessing.py ^
  --train-datasets datasets/02-14-2018.csv datasets/02-15-2018.csv ^
  --epochs 10 ^
  --output-dir artifacts_multi
```

### 3. Обучение на одних днях, тестирование на других

```powershell
.\.venv\Scripts\python.exe preprocessing.py ^
  --train-datasets datasets/02-28-2018.csv ^
  --test-datasets datasets/03-01-2018.csv ^
  --epochs 10 ^
  --output-dir artifacts_crossday
```

Этот режим полезен для более честной оценки модели в дипломе.

## Основные параметры `preprocessing.py`

- `--dataset` - путь к одному CSV.
- `--train-datasets` - список CSV для обучения.
- `--test-datasets` - список CSV для тестирования.
- `--output-dir` - папка для сохранения модели и метаданных.
- `--epochs` - число эпох.
- `--batch-size` - размер батча.
- `--sequence-length` - длина окна для LSTM.
- `--test-fraction` - доля теста при внутридневном split.
- `--validation-fraction` - доля validation внутри train.
- `--patience` - терпение EarlyStopping.
- `--max-rows` - ограничение числа строк для быстрого прогона.

## Что сохраняется после обучения

В папке `output_dir` сохраняются:

- `lstm_ids_model.keras` - обученная модель;
- `scaler.pkl` - объект нормализации;
- `metadata.json` - параметры запуска, mapping классов, размеры выборок, метрики;
- `training_history.png` - графики loss и accuracy по эпохам.

## Эксперименты и сравнение моделей

### Построить профиль классов по всем CSV

```powershell
.\.venv\Scripts\python.exe experiment.py --profile-only
```

Результат:

- `reports/dataset_profile.csv`

### Запустить набор сценариев

```powershell
.\.venv\Scripts\python.exe experiment.py --epochs 1 --fast-max-rows 50000
```

### Запустить только выбранные сценарии

```powershell
.\.venv\Scripts\python.exe experiment.py --scenarios bot_intraday bruteforce_intraday --epochs 10
```

Результат:

- `reports/experiment_summary.csv`
- `reports/<scenario>/...`

Для каждого сценария в сводку попадают модели:

- `lstm`
- `logreg`
- `random_forest`

## Какие сценарии уже есть

В [experiment.py](/C:/Users/Kukakryak/PycharmProjects/ai_shark/experiment.py:29) описаны готовые сценарии:

- `bruteforce_intraday`
- `dos_intraday`
- `slowhttp_vs_hulk_intraday`
- `web_attacks_crossday`
- `infiltration_crossday`
- `bot_intraday`

## Метрики

В проекте используются:

- `precision_macro`
- `recall_macro`
- `f1_macro`
- `classification_report`
- `confusion_matrix`

Для диплома основной упор лучше делать на `macro F1`, потому что датасет несбалансирован и обычная `accuracy` может быть слишком оптимистичной.

## Ограничения проекта

- разные дни CIC-IDS 2018 содержат разные семейства атак;
- не любой `train/test` сценарий методологически корректен;
- при использовании только первых строк файла возможен перекос по классам;
- LSTM не обязана быть лучшей моделью на табличных признаках, поэтому baseline-сравнение обязательно.

## Рекомендуемый порядок работы

1. Проверить окружение и тесты.
2. Запустить один быстрый smoke-run.
3. Запустить основной сценарий обучения.
4. Прогнать `experiment.py` для сравнительной таблицы.
5. Использовать `metadata.json`, `training_history.png` и `experiment_summary.csv` в дипломе.

## Что читать дальше

- [PROJECT_GUIDE.md](/C:/Users/Kukakryak/PycharmProjects/ai_shark/PROJECT_GUIDE.md:1) - если нужно понять весь проект подробно.
- [DIPLOMA_RESULTS.md](/C:/Users/Kukakryak/PycharmProjects/ai_shark/DIPLOMA_RESULTS.md:1) - если нужно оформить результаты.
- [FINAL_EXECUTION_PLAN.md](/C:/Users/Kukakryak/PycharmProjects/ai_shark/FINAL_EXECUTION_PLAN.md:1) - если нужно быстро пройти финальный цикл перед защитой.
