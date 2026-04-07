from __future__ import annotations

import numpy as np

from pipeline import evaluate


class _FakeSingleInputModel:
    def predict(self, inp: np.ndarray, verbose: int = 0) -> np.ndarray:
        assert verbose == 0
        assert inp.shape == (3, 200, 100, 1)
        return np.array(
            [
                [0.1, 0.8, 0.1],
                [0.6, 0.2, 0.2],
                [0.2, 0.2, 0.6],
            ],
            dtype=np.float32,
        )


class _FakeFusionModel:
    def predict(self, inputs: list[np.ndarray], verbose: int = 0) -> np.ndarray:
        x, y = inputs
        assert verbose == 0
        assert x.shape == (2, 200, 100, 1)
        assert y.shape == (2, 200, 100, 1)
        return np.array(
            [
                [0.2, 0.7, 0.1],
                [0.3, 0.1, 0.6],
            ],
            dtype=np.float32,
        )


def test_predict_keras_expands_single_channel_input() -> None:
    features = np.zeros((3, 200, 100), dtype=np.uint8)

    preds = evaluate._predict_keras(_FakeSingleInputModel(), features)

    np.testing.assert_array_equal(preds, np.array([1, 0, 2], dtype=np.int64))


def test_predict_fusion_expands_both_inputs() -> None:
    x_feat = np.zeros((2, 200, 100), dtype=np.uint8)
    y_feat = np.zeros((2, 200, 100), dtype=np.uint8)

    preds = evaluate._predict_fusion(_FakeFusionModel(), x_feat, y_feat)

    np.testing.assert_array_equal(preds, np.array([1, 2], dtype=np.int64))


def test_confusion_matrix_counts_predictions_by_true_label() -> None:
    labels = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    preds = np.array([0, 1, 1, 2, 0], dtype=np.int64)

    cm = evaluate._confusion_matrix(preds, labels, n_classes=3)

    expected = np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(cm, expected)
