import numpy as np

from data.denoise_data import generate_synthetic_paired_dataset
from pipeline.denoise_ingest import prepare_denoising_dataset, prepare_denoising_example


def test_prepare_denoising_example_shapes() -> None:
    clean, _, mixture = generate_synthetic_paired_dataset(n_clips=1, duration_s=0.05, seed=3)
    noisy_inputs, clean_target = prepare_denoising_example(mixture[0], clean[0])
    assert noisy_inputs.shape == (200, 2)
    assert clean_target.shape == (200, 1)


def test_prepare_denoising_dataset_alignment() -> None:
    clean, _, mixture = generate_synthetic_paired_dataset(n_clips=3, duration_s=0.05, seed=4)
    noisy_inputs, clean_targets = prepare_denoising_dataset(mixture, clean)
    assert noisy_inputs.shape == (3, 200, 2)
    assert clean_targets.shape == (3, 200, 1)
    assert np.isfinite(noisy_inputs).all()
    assert np.isfinite(clean_targets).all()
