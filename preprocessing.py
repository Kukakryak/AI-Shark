from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight


@dataclass
class TrainingConfig:
    dataset_path: Path = Path("datasets/02-14-2018.csv")
    train_datasets: tuple[Path, ...] = ()
    test_datasets: tuple[Path, ...] = ()
    output_dir: Path = Path("artifacts")
    sequence_length: int = 10
    test_fraction: float = 0.2
    epochs: int = 30
    batch_size: int = 128
    validation_fraction: float = 0.2
    dropout: float = 0.3
    lstm_units: int = 64
    dense_units: int = 32
    early_stopping_patience: int = 5
    random_seed: int = 42
    max_rows: int | None = None


@dataclass
class PreparedData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    label_column: str
    label_mapping: dict[str, int]
    feature_columns: list[str]
    train_rows: int
    test_rows: int


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def configure_gpu() -> list[str]:
    try:
        import tensorflow as tf
    except Exception:
        return []

    devices = tf.config.list_physical_devices("GPU")
    for device in devices[:1]:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass
    return [device.name for device in devices]


def detect_label_column(dataframe: pd.DataFrame) -> str:
    if "Label" in dataframe.columns:
        return "Label"

    for column in dataframe.columns:
        if pd.api.types.is_object_dtype(dataframe[column]) or pd.api.types.is_string_dtype(dataframe[column]):
            return column

    raise ValueError("Не удалось определить столбец с метками классов.")


def load_dataset(dataset_path: Path, max_rows: int | None = None) -> pd.DataFrame:
    dataframe = pd.read_csv(dataset_path, nrows=max_rows, low_memory=False)
    if "Timestamp" in dataframe.columns:
        dataframe = dataframe.drop(columns=["Timestamp"])
    return dataframe


def load_datasets(dataset_paths: tuple[Path, ...], max_rows: int | None = None) -> pd.DataFrame:
    dataframes = []
    rows_left = max_rows

    for dataset_path in dataset_paths:
        nrows = rows_left if rows_left is not None else None
        dataframe = load_dataset(dataset_path, max_rows=nrows)
        dataframe["SourceFile"] = dataset_path.name
        dataframes.append(dataframe)
        if rows_left is not None:
            rows_left -= len(dataframe)
            if rows_left <= 0:
                break

    if not dataframes:
        raise ValueError("Не удалось загрузить ни одного датасета.")

    return pd.concat(dataframes, ignore_index=True)


def clean_dataframe(dataframe: pd.DataFrame, label_column: str) -> pd.DataFrame:
    cleaned = dataframe.copy()
    cleaned[label_column] = cleaned[label_column].astype(str).str.strip()
    cleaned = cleaned[cleaned[label_column].str.lower() != label_column.lower()]
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("После очистки датасет пуст.")
    if cleaned[label_column].nunique() < 2:
        raise ValueError("Для обучения нужно минимум два класса.")
    return cleaned


def encode_labels(dataframe: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, dict[str, int]]:
    labels = sorted(dataframe[label_column].astype(str).unique())
    mapping = {label: index for index, label in enumerate(labels)}
    encoded = dataframe.copy()
    encoded[label_column] = encoded[label_column].astype(str).map(mapping).astype(int)
    return encoded, mapping


def chronological_split(dataframe: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction должен быть в диапазоне (0, 0.5).")

    split_index = int(len(dataframe) * (1 - test_fraction))
    if split_index <= 1 or split_index >= len(dataframe) - 1:
        raise ValueError("Недостаточно строк для корректного train/test split.")

    train_df = dataframe.iloc[:split_index].reset_index(drop=True)
    test_df = dataframe.iloc[split_index:].reset_index(drop=True)
    return train_df, test_df


def split_by_config(dataframe: pd.DataFrame, config: TrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.train_datasets and config.test_datasets:
        train_names = {path.name for path in config.train_datasets}
        test_names = {path.name for path in config.test_datasets}
        train_df = dataframe[dataframe["SourceFile"].isin(train_names)].reset_index(drop=True)
        test_df = dataframe[dataframe["SourceFile"].isin(test_names)].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            raise ValueError("После разделения по файлам train или test оказались пустыми.")
        return train_df, test_df

    return chronological_split(dataframe, config.test_fraction)


def validate_class_coverage(train_df: pd.DataFrame, test_df: pd.DataFrame, label_column: str) -> None:
    train_labels = set(train_df[label_column].unique())
    test_labels = set(test_df[label_column].unique())
    if len(test_labels) < 2:
        raise ValueError("В test-выборке меньше двух классов. Для дипломной оценки нужно больше разнообразия.")
    missing_in_train = test_labels - train_labels
    if missing_in_train:
        raise ValueError(f"В test есть классы, которых нет в train: {sorted(missing_in_train)}")


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler, list[str]]:
    feature_columns = [column for column in train_df.columns if column not in {label_column, "SourceFile"}]
    scaler = MinMaxScaler()

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    test_scaled[feature_columns] = scaler.transform(test_df[feature_columns])
    train_scaled = train_scaled.drop(columns=["SourceFile"], errors="ignore")
    test_scaled = test_scaled.drop(columns=["SourceFile"], errors="ignore")

    return train_scaled, test_scaled, scaler, feature_columns


def build_sequences(dataframe: pd.DataFrame, label_column: str, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    if sequence_length < 2:
        raise ValueError("sequence_length должен быть >= 2.")
    if len(dataframe) < sequence_length:
        raise ValueError("Для построения последовательностей недостаточно строк.")

    features = dataframe.drop(columns=[label_column]).to_numpy(dtype=np.float32)
    labels = dataframe[label_column].to_numpy(dtype=np.int32)

    x_data = []
    y_data = []
    for start_index in range(len(dataframe) - sequence_length + 1):
        end_index = start_index + sequence_length
        x_data.append(features[start_index:end_index])
        y_data.append(labels[end_index - 1])

    return np.asarray(x_data), np.asarray(y_data)


def prepare_data(config: TrainingConfig) -> PreparedData:
    if config.train_datasets and config.test_datasets:
        raw_train_df = load_datasets(config.train_datasets, max_rows=config.max_rows)
        raw_test_df = load_datasets(config.test_datasets, max_rows=config.max_rows)
        label_column = detect_label_column(raw_train_df)
        cleaned_train_df = clean_dataframe(raw_train_df, label_column)
        cleaned_test_df = clean_dataframe(raw_test_df, label_column)
        combined_df = pd.concat([cleaned_train_df, cleaned_test_df], ignore_index=True)
        _, label_mapping = encode_labels(combined_df, label_column)
        train_df = cleaned_train_df.copy()
        test_df = cleaned_test_df.copy()
        train_df[label_column] = train_df[label_column].astype(str).map(label_mapping).astype(int)
        test_df[label_column] = test_df[label_column].astype(str).map(label_mapping).astype(int)
    else:
        dataset_paths = config.train_datasets if config.train_datasets else (config.dataset_path,)
        raw_df = load_datasets(dataset_paths, max_rows=config.max_rows)
        label_column = detect_label_column(raw_df)
        cleaned_df = clean_dataframe(raw_df, label_column)
        encoded_df, label_mapping = encode_labels(cleaned_df, label_column)
        train_df, test_df = split_by_config(encoded_df, config)

    validate_class_coverage(train_df, test_df, label_column)
    train_scaled, test_scaled, scaler, feature_columns = scale_features(train_df, test_df, label_column)
    x_train, y_train = build_sequences(train_scaled, label_column, config.sequence_length)
    x_test, y_test = build_sequences(test_scaled, label_column, config.sequence_length)

    return PreparedData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        label_column=label_column,
        label_mapping=label_mapping,
        feature_columns=feature_columns,
        train_rows=len(train_df),
        test_rows=len(test_df),
    )


def build_model(input_shape: tuple[int, int], num_classes: int, config: TrainingConfig):
    from keras.layers import Dense, Dropout, Input, LSTM
    from keras.models import Sequential

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(config.lstm_units),
            Dropout(config.dropout),
            Dense(config.dense_units, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def train_model(prepared: PreparedData, config: TrainingConfig):
    from keras.callbacks import EarlyStopping

    num_classes = len(prepared.label_mapping)
    model = build_model(
        input_shape=(prepared.x_train.shape[1], prepared.x_train.shape[2]),
        num_classes=num_classes,
        config=config,
    )
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        prepared.x_train,
        prepared.y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_split=config.validation_fraction,
        callbacks=callbacks,
        class_weight=compute_class_weights(prepared.y_train),
        verbose=1,
    )
    return model, history


def evaluate_model(model, prepared: PreparedData) -> dict[str, object]:
    predictions = model.predict(prepared.x_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = prepared.y_test

    metrics = {
        "precision_macro": float(precision_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "classification_report": classification_report(true_labels, predicted_labels, zero_division=0),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels).tolist(),
    }
    return metrics


def plot_history(history, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history.history["accuracy"]) + 1))
    figure = plt.figure(figsize=(12, 4))

    axis_accuracy = figure.add_subplot(1, 2, 1)
    axis_accuracy.plot(epochs, history.history["accuracy"], marker="o", linewidth=2, label="Train Accuracy")
    axis_accuracy.plot(epochs, history.history["val_accuracy"], marker="o", linewidth=2, label="Validation Accuracy")
    axis_accuracy.set_title("Model Accuracy")
    axis_accuracy.set_xlabel("Epoch")
    axis_accuracy.set_ylabel("Accuracy")
    axis_accuracy.set_xticks(epochs)
    axis_accuracy.grid(True, alpha=0.3)
    axis_accuracy.legend()

    axis_loss = figure.add_subplot(1, 2, 2)
    axis_loss.plot(epochs, history.history["loss"], marker="o", linewidth=2, label="Train Loss")
    axis_loss.plot(epochs, history.history["val_loss"], marker="o", linewidth=2, label="Validation Loss")
    axis_loss.set_title("Model Loss")
    axis_loss.set_xlabel("Epoch")
    axis_loss.set_ylabel("Loss")
    axis_loss.set_xticks(epochs)
    axis_loss.grid(True, alpha=0.3)
    axis_loss.legend()

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_artifacts(model, prepared: PreparedData, config: TrainingConfig, metrics: dict[str, object], history) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    model.save(config.output_dir / "lstm_ids_model.keras")
    plot_history(history, config.output_dir / "training_history.png")

    with (config.output_dir / "scaler.pkl").open("wb") as file:
        pickle.dump(prepared.scaler, file)

    serialized_config = asdict(config)
    serialized_config["dataset_path"] = str(config.dataset_path)
    serialized_config["output_dir"] = str(config.output_dir)
    serialized_config["train_datasets"] = [str(path) for path in config.train_datasets]
    serialized_config["test_datasets"] = [str(path) for path in config.test_datasets]

    metadata = {
        "config": serialized_config,
        "label_column": prepared.label_column,
        "feature_columns": prepared.feature_columns,
        "label_mapping": prepared.label_mapping,
        "train_rows": prepared.train_rows,
        "test_rows": prepared.test_rows,
        "train_sequences": int(len(prepared.x_train)),
        "test_sequences": int(len(prepared.x_test)),
        "metrics": metrics,
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }
    (config.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(prepared: PreparedData, metrics: dict[str, object], gpu_devices: list[str]) -> None:
    print("=" * 80)
    print("Система интеллектуального анализа сетевого трафика")
    print("=" * 80)
    print(f"GPU devices: {gpu_devices or ['CPU only']}")
    print(f"Train rows: {prepared.train_rows}, test rows: {prepared.test_rows}")
    print(f"Train sequences: {len(prepared.x_train)}, test sequences: {len(prepared.x_test)}")
    print(f"Classes: {prepared.label_mapping}")
    print()
    print(metrics["classification_report"])
    print("Confusion matrix:")
    print(np.asarray(metrics["confusion_matrix"]))
    print(f"Precision macro: {metrics['precision_macro']:.4f}")
    print(f"Recall macro: {metrics['recall_macro']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Обучение LSTM-модели для выявления сетевых атак.")
    parser.add_argument("--dataset", default=str(TrainingConfig.dataset_path), help="Путь к CSV-файлу.")
    parser.add_argument("--train-datasets", nargs="+", default=None, help="Список CSV-файлов для обучения.")
    parser.add_argument("--test-datasets", nargs="+", default=None, help="Список CSV-файлов для тестирования.")
    parser.add_argument("--output-dir", default=str(TrainingConfig.output_dir), help="Каталог для артефактов.")
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--sequence-length", type=int, default=TrainingConfig.sequence_length)
    parser.add_argument("--test-fraction", type=float, default=TrainingConfig.test_fraction)
    parser.add_argument("--validation-fraction", type=float, default=TrainingConfig.validation_fraction)
    parser.add_argument("--patience", type=int, default=TrainingConfig.early_stopping_patience)
    parser.add_argument("--max-rows", type=int, default=None, help="Ограничить число строк для быстрой отладки.")
    args = parser.parse_args()

    return TrainingConfig(
        dataset_path=Path(args.dataset),
        train_datasets=tuple(Path(path) for path in (args.train_datasets or [])),
        test_datasets=tuple(Path(path) for path in (args.test_datasets or [])),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
        early_stopping_patience=args.patience,
        max_rows=args.max_rows,
    )


def run_training(config: TrainingConfig) -> dict[str, object]:
    set_global_seed(config.random_seed)
    gpu_devices = configure_gpu()
    prepared = prepare_data(config)
    model, history = train_model(prepared, config)
    metrics = evaluate_model(model, prepared)
    save_artifacts(model, prepared, config, metrics, history)
    print_summary(prepared, metrics, gpu_devices)
    return metrics


def main() -> int:
    config = parse_args()
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
