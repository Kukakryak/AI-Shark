import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import (
    TrainingConfig,
    build_sequences,
    chronological_split,
    clean_dataframe,
    detect_label_column,
    encode_labels,
    split_by_config,
    validate_class_coverage,
)


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.DataFrame(
            {
                "FeatureA": [1.0, 2.0, 3.0, 4.0, np.inf, 6.0],
                "FeatureB": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "Label": ["Benign", "DoS", "Benign", "Bot", "Bot", "DoS"],
            }
        )

    def test_detect_label_column(self) -> None:
        self.assertEqual(detect_label_column(self.dataframe), "Label")

    def test_clean_dataframe_removes_invalid_rows(self) -> None:
        cleaned = clean_dataframe(self.dataframe, "Label")
        self.assertEqual(len(cleaned), 5)
        self.assertFalse(np.isinf(cleaned.drop(columns=["Label"]).to_numpy()).any())

    def test_encode_labels_is_deterministic(self) -> None:
        cleaned = clean_dataframe(self.dataframe, "Label")
        encoded, mapping = encode_labels(cleaned, "Label")
        self.assertEqual(mapping, {"Benign": 0, "Bot": 1, "DoS": 2})
        self.assertTrue(pd.api.types.is_integer_dtype(encoded["Label"]))

    def test_chronological_split_preserves_order(self) -> None:
        cleaned = clean_dataframe(self.dataframe, "Label")
        train_df, test_df = chronological_split(cleaned, 0.4)
        self.assertEqual(train_df.iloc[0]["FeatureA"], 1.0)
        self.assertEqual(test_df.iloc[0]["FeatureA"], 4.0)

    def test_build_sequences_shapes(self) -> None:
        encoded, _ = encode_labels(clean_dataframe(self.dataframe, "Label"), "Label")
        x_data, y_data = build_sequences(encoded, "Label", sequence_length=3)
        self.assertEqual(x_data.shape, (3, 3, 2))
        self.assertEqual(y_data.shape, (3,))

    def test_split_by_config_uses_source_files(self) -> None:
        dataframe = pd.DataFrame(
            {
                "FeatureA": [1.0, 2.0, 3.0, 4.0],
                "Label": [0, 1, 0, 1],
                "SourceFile": ["a.csv", "a.csv", "b.csv", "b.csv"],
            }
        )
        config = TrainingConfig(
            train_datasets=(Path("a.csv"),),
            test_datasets=(Path("b.csv"),),
        )
        train_df, test_df = split_by_config(dataframe, config)
        self.assertEqual(len(train_df), 2)
        self.assertEqual(len(test_df), 2)

    def test_validate_class_coverage_rejects_single_class_test(self) -> None:
        train_df = pd.DataFrame({"Label": [0, 1, 0]})
        test_df = pd.DataFrame({"Label": [1, 1]})
        with self.assertRaises(ValueError):
            validate_class_coverage(train_df, test_df, "Label")


if __name__ == "__main__":
    unittest.main()
