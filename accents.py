"""Train, evaluate and apply a multiple-kernel ELM accent classifier."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel

from features import extract


def read_manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        keys = ("path", "accent", "speaker")
        if not set(keys).issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires path,accent,speaker columns")
        rows = []
        seen = set()
        for row in reader:
            if any(not (row.get(k) or "").strip() for k in keys):
                raise ValueError(f"Missing value on line {reader.line_num}")
            audio = (path.parent / row["path"].strip()).resolve()
            if audio in seen:
                raise ValueError(f"Duplicate recording: {audio}")
            seen.add(audio)
            rows.append((audio, row["accent"].strip(), row["speaker"].strip()))
    if not rows:
        raise ValueError("Empty manifest")
    return zip(*rows)


def kernel(x, reference, weight):
    linear = x @ reference.T / x.shape[1]
    return (1 - weight) * linear + weight * rbf_kernel(x, reference, gamma=1 / x.shape[1])


def train(args):
    paths, labels, speakers = read_manifest(args.manifest)
    classes, indices = np.unique(labels, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("Training requires at least two accent labels")
    if not (np.isfinite(args.c) and args.c > 0 and 0 <= args.rbf_weight <= 1):
        raise ValueError("C must be finite and positive; RBF weight must be in [0, 1]")
    features = np.stack([extract(path) for path in paths])
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale < 1e-8] = 1
    reference = (features - mean) / scale
    gram = kernel(reference, reference, args.rbf_weight)
    gram.flat[::len(gram) + 1] += 1 / args.c
    beta = solve(gram, np.eye(len(classes))[indices], assume_a="pos")
    with args.model.open("wb") as stream:
        np.savez_compressed(stream, mean=mean, scale=scale, reference=reference,
                            weight=args.rbf_weight, beta=beta, classes=classes,
                            speakers=np.unique(speakers), paths=[str(p) for p in paths])
    print(f"Saved {args.model} from {len(paths)} recordings and {len(set(speakers))} speakers")


def predict(model, paths):
    features = np.stack([extract(path) for path in paths])
    x = (features - model["mean"]) / model["scale"]
    scores = kernel(x, model["reference"], model["weight"]) @ model["beta"]
    return model["classes"][scores.argmax(axis=1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("manifest", type=Path)
    fit.add_argument("model", type=Path)
    fit.add_argument("--c", type=float, default=10.0)
    fit.add_argument("--rbf-weight", type=float, default=0.5)
    test = commands.add_parser("evaluate")
    test.add_argument("manifest", type=Path)
    test.add_argument("model", type=Path)
    infer = commands.add_parser("predict")
    infer.add_argument("audio", type=Path)
    infer.add_argument("model", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
        return
    with np.load(args.model, allow_pickle=False) as model:
        if args.command == "predict":
            print(predict(model, [args.audio])[0])
            return
        paths, labels, speakers = read_manifest(args.manifest)
        if set(speakers).intersection(model["speakers"]) or set(map(str, paths)).intersection(model["paths"]):
            raise ValueError("Evaluation overlaps training speakers or recording paths")
        classes = model["classes"]
        if set(labels) != set(classes):
            raise ValueError("Evaluation must contain every trained accent and no unseen labels")
        predictions = predict(model, paths)
        print(f"Utterance accuracy: {accuracy_score(labels, predictions):.4f}")
        print(classification_report(labels, predictions, labels=classes, digits=4, zero_division=0))
        print("Confusion matrix (rows=true, columns=predicted):", classes.tolist())
        print(confusion_matrix(labels, predictions, labels=classes))


if __name__ == "__main__":
    main()
