"""Convert a WAV utterance to 120 MFCC summary features."""
from math import gcd

import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import resample_poly


def extract(path):
    rate, samples = wavfile.read(path)
    audio = samples.astype(np.float64)
    if samples.dtype == np.uint8:
        audio -= 128
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"Empty or nonfinite recording: {path}")
    audio -= audio.mean()
    peak = np.max(np.abs(audio))
    if peak == 0:
        raise ValueError(f"Silent recording: {path}")
    audio /= peak
    divisor = gcd(rate, 16000)
    audio = resample_poly(audio, 16000 // divisor, rate // divisor)
    audio = np.pad(audio, (0, max(0, 400 - len(audio))))
    frames = np.lib.stride_tricks.sliding_window_view(audio, 400)[::160]
    power = abs(np.fft.rfft(frames * np.hamming(400), n=512)) ** 2
    hz = np.fft.rfftfreq(512, 1 / 16000)
    edges = 700 * (np.geomspace(1 + 50 / 700, 1 + 7600 / 700, 42) - 1)
    left = (hz - edges[:-2, None]) / np.diff(edges)[:-1, None]
    right = (edges[2:, None] - hz) / np.diff(edges)[1:, None]
    bank = np.maximum(0, np.minimum(left, right))
    log_mel = np.log(np.maximum(power @ bank.T, 1e-10))
    static = dct(log_mel, type=2, norm="ortho", axis=1)[:, :20]

    def delta(sequence):
        padded = np.pad(sequence, ((1, 1), (0, 0)), mode="edge")
        return (padded[2:] - padded[:-2]) / 2

    first = delta(static)
    frames = np.hstack([static, first, delta(first)])
    return np.r_[frames.mean(axis=0), frames.std(axis=0)]
