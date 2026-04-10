import pytest

tf = pytest.importorskip("tensorflow")

from data.denoise_data import generate_synthetic_paired_dataset
from pipeline.denoise_ingest import prepare_denoising_dataset
from pipeline.models.denoising.tcn_denoiser import TCNDenoiser


def test_tiny_denoiser_training_improves_over_noisy_baseline(tmp_path) -> None:
    clean, _, mixture = generate_synthetic_paired_dataset(
        n_clips=8, duration_s=0.05, seed=5,
    )
    noisy_inputs, clean_targets = prepare_denoising_dataset(mixture, clean)

    denoiser = TCNDenoiser(epochs=2, batch_size=4, base_channels=16)
    checkpoint_path = tmp_path / "denoiser.keras"
    denoiser.fit(
        noisy_inputs,
        clean_targets,
        validation_data=(noisy_inputs, clean_targets),
        checkpoint_path=checkpoint_path,
    )
    preds = denoiser.predict(noisy_inputs)

    assert preds.shape == clean_targets.shape
    assert checkpoint_path.exists()
    assert denoiser.history_["loss"][-1] <= denoiser.history_["loss"][0]
    assert denoiser.history_["val_loss"][-1] <= denoiser.history_["val_loss"][0]
