from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from preprocessing import (
    TrainingConfig,
    clean_dataframe,
    detect_label_column,
    load_dataset,
    prepare_data,
    run_training,
)


@dataclass(frozen=True)
class ExperimentScenario:
    name: str
    description: str
    train_datasets: tuple[str, ...] = ()
    test_datasets: tuple[str, ...] = ()
    dataset: str | None = None
    max_rows: int | None = None


DATASET_DIR = Path("datasets")
REPORTS_DIR = Path("reports")


SCENARIOS: tuple[ExperimentScenario, ...] = (
    ExperimentScenario(
        name="bruteforce_intraday",
        description="FTP/SSH brute force внутри 02-14-2018, хронологический split.",
        train_datasets=("02-14-2018.csv",),
        max_rows=120000,
    ),
    ExperimentScenario(
        name="dos_intraday",
        description="DoS-атаки внутри 02-15-2018, хронологический split.",
        train_datasets=("02-15-2018.csv",),
        max_rows=120000,
    ),
    ExperimentScenario(
        name="slowhttp_vs_hulk_intraday",
        description="Смешанный DoS день 02-16-2018, хронологический split.",
        train_datasets=("02-16-2018.csv",),
        max_rows=150000,
    ),
    ExperimentScenario(
        name="web_attacks_crossday",
        description="Обучение на 02-22-2018, тестирование на 02-23-2018 для Web/XSS/SQL.",
        train_datasets=("02-22-2018.csv",),
        test_datasets=("02-23-2018.csv",),
        max_rows=None,
    ),
    ExperimentScenario(
        name="infiltration_crossday",
        description="Обучение на 02-28-2018, тестирование на 03-01-2018 для Infilteration.",
        train_datasets=("02-28-2018.csv",),
        test_datasets=("03-01-2018.csv",),
        max_rows=None,
    ),
    ExperimentScenario(
        name="bot_intraday",
        description="Bot-атаки внутри 03-02-2018, хронологический split.",
        train_datasets=("03-02-2018.csv",),
        max_rows=120000,
    ),
)


def profile_datasets(output_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_path in sorted(DATASET_DIR.glob("*.csv")):
        dataframe = pd.read_csv(dataset_path, usecols=["Label"])
        dataframe["Label"] = dataframe["Label"].astype(str).str.strip()
        cleaned = dataframe[dataframe["Label"].str.lower() != "label"].reset_index(drop=True)
        counts = cleaned["Label"].value_counts()
        total_rows = int(len(cleaned))
        for label, count in counts.items():
            rows.append(
                {
                    "dataset": dataset_path.name,
                    "label": str(label),
                    "count": int(count),
                    "share": round(count / total_rows, 6),
                    "total_rows": total_rows,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["dataset", "label", "count", "share", "total_rows"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_config(scenario: ExperimentScenario, output_dir: Path, epochs: int, batch_size: int) -> TrainingConfig:
    return TrainingConfig(
        dataset_path=DATASET_DIR / scenario.train_datasets[0] if scenario.train_datasets else Path(scenario.dataset or ""),
        train_datasets=tuple(DATASET_DIR / name for name in scenario.train_datasets),
        test_datasets=tuple(DATASET_DIR / name for name in scenario.test_datasets),
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        max_rows=scenario.max_rows,
    )


def flatten_sequences(x_data):
    return x_data.reshape(x_data.shape[0], -1)


def evaluate_predictions(y_true, y_pred) -> dict[str, object]:
    return {
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def run_baselines(config: TrainingConfig, output_dir: Path) -> list[dict[str, object]]:
    prepared = prepare_data(config)
    x_train = flatten_sequences(prepared.x_train)
    x_test = flatten_sequences(prepared.x_test)
    y_train = prepared.y_train
    y_test = prepared.y_test

    baselines = [
        (
            "logreg",
            LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                random_state=config.random_seed,
                n_jobs=None,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=None,
                class_weight="balanced_subsample",
                random_state=config.random_seed,
                n_jobs=-1,
            ),
        ),
    ]

    rows: list[dict[str, object]] = []
    baseline_dir = output_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    for baseline_name, model in baselines:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = evaluate_predictions(y_test, predictions)
        metadata = {
            "model": baseline_name,
            "train_sequences": int(len(prepared.x_train)),
            "test_sequences": int(len(prepared.x_test)),
            "label_mapping": prepared.label_mapping,
            "metrics": metrics,
        }
        (baseline_dir / f"{baseline_name}_metrics.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        rows.append(
            {
                "model": baseline_name,
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "train_sequences": int(len(prepared.x_train)),
                "test_sequences": int(len(prepared.x_test)),
                "classes": ", ".join(prepared.label_mapping.keys()),
            }
        )

    return rows


def run_scenarios(selected_names: set[str] | None, epochs: int, batch_size: int, fast_max_rows: int | None) -> pd.DataFrame:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for scenario in SCENARIOS:
        if selected_names and scenario.name not in selected_names:
            continue

        output_dir = REPORTS_DIR / scenario.name
        effective_max_rows = fast_max_rows if fast_max_rows is not None else scenario.max_rows
        try:
            config = build_config(scenario, output_dir, epochs, batch_size)
            config.max_rows = effective_max_rows
            metrics = run_training(config)
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "scenario": scenario.name,
                    "model": "lstm",
                    "description": scenario.description,
                    "status": "ok",
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                    "f1_macro": metrics["f1_macro"],
                    "train_sequences": metadata["train_sequences"],
                    "test_sequences": metadata["test_sequences"],
                    "classes": ", ".join(metadata["label_mapping"].keys()),
                }
            )
            for baseline_row in run_baselines(config, output_dir):
                rows.append(
                    {
                        "scenario": scenario.name,
                        "model": baseline_row["model"],
                        "description": scenario.description,
                        "status": "ok",
                        "precision_macro": baseline_row["precision_macro"],
                        "recall_macro": baseline_row["recall_macro"],
                        "f1_macro": baseline_row["f1_macro"],
                        "train_sequences": baseline_row["train_sequences"],
                        "test_sequences": baseline_row["test_sequences"],
                        "classes": baseline_row["classes"],
                    }
                )
        except Exception as error:
            rows.append(
                {
                    "scenario": scenario.name,
                    "model": "pipeline",
                    "description": scenario.description,
                    "status": "failed",
                    "precision_macro": None,
                    "recall_macro": None,
                    "f1_macro": None,
                    "train_sequences": None,
                    "test_sequences": None,
                    "classes": str(error),
                }
            )

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(REPORTS_DIR / "experiment_summary.csv", index=False, encoding="utf-8")
    return dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск набора экспериментальных сценариев для диплома.")
    parser.add_argument("--profile-only", action="store_true", help="Только построить профиль классов по датасетам.")
    parser.add_argument("--scenarios", nargs="+", default=None, help="Список сценариев для запуска.")
    parser.add_argument("--epochs", type=int, default=1, help="Число эпох для каждого сценария.")
    parser.add_argument("--batch-size", type=int, default=128, help="Размер батча.")
    parser.add_argument("--fast-max-rows", type=int, default=None, help="Переопределить max_rows для всех сценариев.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile_path = REPORTS_DIR / "dataset_profile.csv"
    profile_datasets(profile_path)
    print(f"Dataset profile saved to {profile_path}")

    if args.profile_only:
        return 0

    summary = run_scenarios(
        set(args.scenarios) if args.scenarios else None,
        args.epochs,
        args.batch_size,
        args.fast_max_rows,
    )
    print(summary.to_string(index=False))
    print(f"\nExperiment summary saved to {REPORTS_DIR / 'experiment_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
