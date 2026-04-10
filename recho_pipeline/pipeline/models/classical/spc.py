"""
SPC Monitor — Statistical Process Control anomaly detector.

The simplest possible real-time anomaly detector for Hopf oscillator output.
Operates on the orbit radius r(t) = sqrt(x^2 + y^2) computed sample-by-sample.

This is the model to deploy first on M33 — no ML, no quantisation, just
arithmetic. Runs in < 10 clock cycles per sample on Cortex-M33.

Firmware equivalent:
    float r = sqrtf(x*x + y*y);   // arm_sqrt_f32() on M33
    // Compare to precomputed UCL/LCL thresholds stored in flash

Western Electric rules implemented:
    Rule 1: 1 point beyond 3-sigma (most common fault signature)
    Rule 2: 9 consecutive points on same side of mean
    Rule 3: 6 consecutive points trending up or down
    Rule 4: 2 of 3 consecutive points beyond 2-sigma on same side

Reference:
  Western Electric Statistical Quality Control Handbook (1956)
  Shougat et al., Scientific Reports 2021 — paper 1
    orbit radius is the natural health metric for Hopf limit cycle monitoring
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))


class SPCMonitor:
    """
    Real-time Statistical Process Control monitor for Hopf oscillator orbit radius.

    Computes r(t) = sqrt(x^2 + y^2) per ADC sample and checks Western Electric
    rules against a baseline fitted on normal operation data.

    This directly monitors the limit cycle stability of the Hopf oscillator —
    when the input signal is anomalous, r(t) deviates from its normal range.

    Example:
        monitor = SPCMonitor(sigma_n=3)
        monitor.fit(x_normal, y_normal)
        result = monitor.update(x_sample, y_sample)
        if result["anomaly"]:
            print(f"Rule {result['rule_violated']} at {result['sigma_level']:.1f}σ")
    """

    def __init__(self, sigma_n: float = 3.0) -> None:
        """
        Args:
            sigma_n: number of standard deviations for UCL/LCL (default 3).
        """
        self.sigma_n = sigma_n
        self._mean: float = 0.0
        self._std: float = 1.0
        self._ucl: float = 3.0       # Upper Control Limit (mean + sigma_n * std)
        self._lcl: float = -3.0      # Lower Control Limit (mean - sigma_n * std)
        self._ucl_2s: float = 2.0    # 2-sigma upper limit (for Rule 4)
        self._lcl_2s: float = -2.0   # 2-sigma lower limit (for Rule 4)
        self._is_fitted: bool = False

        # Sliding window buffers for Western Electric rules
        self._recent: deque[float] = deque(maxlen=9)    # last 9 radii
        self._side_buffer: deque[int] = deque(maxlen=9) # sign of (r - mean)
        self._trend_buffer: deque[float] = deque(maxlen=6)  # last 6 for trend

    def fit(
        self,
        x_normal: NDArray[np.float64],
        y_normal: NDArray[np.float64],
    ) -> "SPCMonitor":
        """
        Compute baseline statistics from normal operation data.

        Fits control limits using mean and std of orbit radius r(t) = sqrt(x^2+y^2).
        Both x_normal and y_normal should be 1-D arrays of time-domain samples
        from the Hopf oscillator during normal (fault-free) operation.

        Args:
            x_normal: 1-D array of x(t) samples during normal operation
            y_normal: 1-D array of y(t) samples, same length as x_normal

        Returns:
            self (for chaining)
        """
        r = np.sqrt(x_normal ** 2 + y_normal ** 2)
        self._mean = float(np.mean(r))
        self._std = float(np.std(r))
        if self._std < 1e-12:
            self._std = 1.0

        self._ucl = self._mean + self.sigma_n * self._std
        self._lcl = self._mean - self.sigma_n * self._std
        self._ucl_2s = self._mean + 2.0 * self._std
        self._lcl_2s = self._mean - 2.0 * self._std
        self._is_fitted = True

        print(f"[SPCMonitor] Fitted: mean={self._mean:.4f}, std={self._std:.4f}, "
              f"UCL={self._ucl:.4f}, LCL={self._lcl:.4f}")
        return self

    def update(
        self,
        x_sample: float,
        y_sample: float,
    ) -> dict:
        """
        Process one ADC sample pair. Called once per sample in firmware.

        Computes r = sqrt(x^2 + y^2) and checks all four Western Electric rules.

        Args:
            x_sample: current x(t) ADC reading
            y_sample: current y(t) ADC reading

        Returns:
            dict with keys:
              "anomaly"       — bool: True if any rule was violated
              "radius"        — float: current orbit radius r
              "sigma_level"   — float: how many sigmas from mean
              "rule_violated" — str: which rule fired, or "" if none
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before update()")

        # Core computation: orbit radius — maps to arm_sqrt_f32() on M33
        r = float(np.sqrt(x_sample ** 2 + y_sample ** 2))
        sigma_level = abs(r - self._mean) / self._std if self._std > 0 else 0.0

        self._recent.append(r)
        self._side_buffer.append(1 if r >= self._mean else -1)
        self._trend_buffer.append(r)

        rule_violated = ""

        # Rule 1: 1 point beyond 3-sigma (UCL or LCL)
        if r > self._ucl or r < self._lcl:
            rule_violated = "Rule1_3sigma"

        # Rule 2: 9 consecutive points on same side of mean
        elif len(self._side_buffer) == 9 and abs(sum(self._side_buffer)) == 9:
            rule_violated = "Rule2_9_same_side"

        # Rule 3: 6 consecutive points trending (monotonically) up or down
        elif len(self._trend_buffer) == 6:
            diffs = [self._trend_buffer[i + 1] - self._trend_buffer[i]
                     for i in range(5)]
            if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
                rule_violated = "Rule3_6_trending"

        # Rule 4: 2 of 3 consecutive points beyond 2-sigma on same side
        elif len(self._recent) >= 3:
            last3 = list(self._recent)[-3:]
            above_2s = sum(1 for v in last3 if v > self._ucl_2s)
            below_2s = sum(1 for v in last3 if v < self._lcl_2s)
            if above_2s >= 2 or below_2s >= 2:
                rule_violated = "Rule4_2of3_2sigma"

        return {
            "anomaly": rule_violated != "",
            "radius": r,
            "sigma_level": sigma_level,
            "rule_violated": rule_violated,
        }

    def process_stream(
        self,
        x_stream: NDArray[np.float64],
        y_stream: NDArray[np.float64],
    ) -> list[dict]:
        """
        Process an entire time series stream, returning one result dict per sample.

        Args:
            x_stream: 1-D array of x(t) samples
            y_stream: 1-D array of y(t) samples, same length

        Returns:
            list of result dicts from update()
        """
        return [self.update(float(x), float(y))
                for x, y in zip(x_stream, y_stream)]

    @property
    def control_limits(self) -> dict[str, float]:
        """Return fitted control limits."""
        return {
            "mean": self._mean, "std": self._std,
            "UCL": self._ucl, "LCL": self._lcl,
            "UCL_2s": self._ucl_2s, "LCL_2s": self._lcl_2s,
        }


def main() -> None:
    """Demonstrate SPCMonitor on synthetic Hopf oscillator data."""
    import matplotlib.pyplot as plt
    from data.sample_data import generate_dataset_xy, CLASS_NAMES

    print("[spc] Generating synthetic data ...")
    raw_x, raw_y, labels = generate_dataset_xy(
        n_clips_per_class=5, n_classes=5, cache=False,
    )

    # Downsample to 4 kHz: factor 25
    ds_factor = 25
    x_ds = raw_x[:, ::ds_factor]  # (n_clips, 4000)
    y_ds = raw_y[:, ::ds_factor]

    # Fit on class 0 (sine — treat as "normal operation")
    normal_mask = labels == 0
    x_normal_all = x_ds[normal_mask].flatten()
    y_normal_all = y_ds[normal_mask].flatten()

    monitor = SPCMonitor(sigma_n=3.0)
    monitor.fit(x_normal_all, y_normal_all)

    print("\nControl limits:")
    for k, v in monitor.control_limits.items():
        print(f"  {k}: {v:.4f}")

    # Test on each class
    print("\nAnomaly detection per class:")
    for cls in range(5):
        mask = labels == cls
        clip_x = x_ds[mask][0]
        clip_y = y_ds[mask][0]
        results = monitor.process_stream(clip_x, clip_y)
        n_anomaly = sum(r["anomaly"] for r in results)
        frac = n_anomaly / len(results) * 100
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): {n_anomaly}/{len(results)} "
              f"anomalous samples ({frac:.1f}%)")

    print("[spc] Done.")


if __name__ == "__main__":
    main()
