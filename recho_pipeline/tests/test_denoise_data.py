import numpy as np

from data.denoise_data import generate_synthetic_paired_dataset, train_val_test_split


def test_generate_synthetic_paired_dataset_shapes() -> None:
    clean, noise, mixture = generate_synthetic_paired_dataset(
        n_clips=6, duration_s=0.1, seed=1,
    )
    assert clean.shape == noise.shape == mixture.shape
    assert clean.shape[0] == 6
    assert clean.shape[1] == 400
    assert np.max(np.abs(mixture)) <= 0.95 + 1e-6


def test_train_val_test_split_preserves_counts() -> None:
    clean, noise, mixture = generate_synthetic_paired_dataset(
        n_clips=20, duration_s=0.05, seed=2,
    )
    splits = train_val_test_split(clean, noise, mixture, val_fraction=0.2, test_fraction=0.1, seed=0)
    total = sum(len(splits[name]["clean"]) for name in ("train", "val", "test"))
    assert total == 20
    for subset in ("train", "val", "test"):
        assert splits[subset]["clean"].shape == splits[subset]["mixture"].shape
