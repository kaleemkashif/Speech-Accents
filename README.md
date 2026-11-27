# Speech Accents

Train a CPU-based multiple-kernel extreme learning machine (MKELM) to identify the accent label of a speech recording. The label vocabulary comes from your training corpus. The project includes training, speaker-independent evaluation and single-file prediction.

## Quick start

Use Python 3.10+ and PCM or floating-point WAV recordings. Stereo files are mixed to mono. Prepare separate training and evaluation CSV manifests with the same header. Paths may be absolute or relative to the CSV. Use a consistent, globally unique ID per speaker across both manifests.

```csv
path,accent,speaker
audio/utterance_001.wav,Scottish,spk001
audio/utterance_002.wav,Irish,spk002
```

Include several speakers per accent. Put all recordings from one speaker in the same partition. Include every training accent in the evaluation set. The evaluator checks for shared speaker IDs and recording paths.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python accents.py train train.csv model.npz
python accents.py evaluate test.csv model.npz
python accents.py predict sample.wav model.npz
```

## Model

Audio is resampled to 16 kHz and analyzed in 25 ms windows every 10 ms. Forty mel filters feed 20 MFCCs. Means and standard deviations of the static, delta and delta-delta coefficients give 120 features per utterance. The training set supplies the feature means and scales used for all subsequent recordings.

The kernel is a weighted sum of a dimension-scaled linear kernel and a Gaussian RBF kernel. Training solves `(K + I/C) B = Y` for one-hot accent targets. Prediction selects the largest entry of `K_test,train B`. The default RBF weight is 0.5 and C is 10. Use `--rbf-weight` and `--c` at training time to compare configurations on separate validation speakers. A weight of 0 or 1 gives a single-kernel baseline. The exact solver uses quadratic training memory.

Evaluation prints utterance accuracy, per-accent precision, recall and F1, macro averages and a labeled confusion matrix. Predictions are closed-set and always select one trained accent. A labeled speech corpus is required to measure real-world accuracy. The repository includes source code only.

Related reading: [MKELM based multi-classification model for foreign accent identification](https://pmc.ncbi.nlm.nih.gov/articles/PMC11388364/). This project implements an independent baseline with its own feature and kernel choices.
