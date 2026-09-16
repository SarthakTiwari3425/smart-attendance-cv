"""Offline evaluation of the recognition model.

Attendance is only as trustworthy as the recogniser behind it, so the project
ships a measurement path as a first-class feature rather than an afterthought.

Metrics produced:

* **Rank-1 accuracy** on a held-out, stratified split
* **Per-student precision / recall / F1**
* **Confusion matrix** (saved as CSV and as a PNG heat-map)
* **Threshold sweep** - accuracy, false-accept and false-reject rates across
  candidate distance thresholds, which is how ``distance_threshold`` in
  ``config.yaml`` should be chosen for a new cohort or camera.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dataset import DatasetSplit
from .logger import get_logger
from .recognizer import FaceRecognizer

logger = get_logger(__name__)


@dataclass
class ClassMetrics:
    student_id: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass
class EvaluationResult:
    """Aggregate outcome of one evaluation run."""

    algorithm: str
    threshold: float
    n_train: int
    n_test: int
    n_classes: int
    accuracy: float
    accuracy_with_threshold: float
    false_reject_rate: float
    mean_distance_correct: float
    mean_distance_wrong: float
    per_class: List[ClassMetrics] = field(default_factory=list)
    confusion: List[List[int]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    threshold_sweep: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["per_class"] = [asdict(c) for c in self.per_class]
        return payload

    def summary_lines(self) -> List[str]:
        return [
            f"Algorithm                : {self.algorithm}",
            f"Students / test samples  : {self.n_classes} / {self.n_test}",
            f"Rank-1 accuracy          : {self.accuracy * 100:.2f}%",
            f"Accuracy @ threshold {self.threshold:g} : "
            f"{self.accuracy_with_threshold * 100:.2f}%",
            f"False-reject rate        : {self.false_reject_rate * 100:.2f}%",
            f"Mean distance (correct)  : {self.mean_distance_correct:.2f}",
            f"Mean distance (wrong)    : {self.mean_distance_wrong:.2f}",
        ]


def evaluate(
    recognizer: FaceRecognizer,
    test: DatasetSplit,
    threshold: Optional[float] = None,
    n_train: int = 0,
) -> EvaluationResult:
    """Score a trained recogniser on a held-out split."""
    if len(test) == 0:
        raise ValueError("Test split is empty")
    threshold = (
        float(threshold)
        if threshold is not None
        else float(recognizer.config.distance_threshold)
    )

    labels_sorted = sorted(set(test.labels) | set(recognizer.label_map))
    index_of = {label: i for i, label in enumerate(labels_sorted)}
    size = len(labels_sorted)
    confusion = np.zeros((size, size), dtype=int)

    correct = 0
    correct_thresholded = 0
    rejected_correct = 0
    distances_correct: List[float] = []
    distances_wrong: List[float] = []
    records: List[Tuple[int, int, float]] = []   # (true, predicted, distance)

    for image, true_label in zip(test.images, test.labels):
        predicted, distance = recognizer.predict_raw(image)
        records.append((true_label, predicted, distance))
        confusion[index_of[true_label], index_of.get(predicted, 0)] += 1
        if predicted == true_label:
            correct += 1
            distances_correct.append(distance)
            if distance <= threshold:
                correct_thresholded += 1
            else:
                rejected_correct += 1
        else:
            distances_wrong.append(distance)

    total = len(test)
    per_class = _per_class_metrics(confusion, labels_sorted, test.label_map)

    result = EvaluationResult(
        algorithm=recognizer.config.algorithm,
        threshold=threshold,
        n_train=n_train,
        n_test=total,
        n_classes=size,
        accuracy=correct / total,
        accuracy_with_threshold=correct_thresholded / total,
        false_reject_rate=rejected_correct / total,
        mean_distance_correct=float(np.mean(distances_correct)) if distances_correct else 0.0,
        mean_distance_wrong=float(np.mean(distances_wrong)) if distances_wrong else 0.0,
        per_class=per_class,
        confusion=confusion.tolist(),
        labels=[test.label_map.get(l, recognizer.label_map.get(l, str(l))) for l in labels_sorted],
        threshold_sweep=sweep_thresholds(records),
    )
    logger.info("Evaluation complete: accuracy %.2f%%", result.accuracy * 100)
    return result


def calibrate_threshold(
    recognizer: FaceRecognizer, holdout: DatasetSplit, sigma: float = 3.0
) -> float:
    """Derive a rejection threshold from the genuine-match distribution.

    The absolute scale of an LBPH distance depends on image size, grid layout
    and camera, so a hard-coded threshold rarely transfers between datasets.
    Instead we measure the distances of *correct* matches on held-out samples
    and place the cut-off at ``mean + sigma * std`` - far enough out to accept
    almost every genuine face, close enough to reject impostors whose
    distances sit well above the genuine cluster.
    """
    genuine: List[float] = []
    for image, true_label in zip(holdout.images, holdout.labels):
        predicted, distance = recognizer.predict_raw(image)
        if predicted == true_label:
            genuine.append(distance)
    if not genuine:
        logger.warning("Calibration found no correct matches; keeping configured threshold")
        return float(recognizer.config.distance_threshold)
    threshold = float(np.mean(genuine) + sigma * np.std(genuine))
    logger.info(
        "Calibrated threshold: %.2f (genuine mean %.2f, std %.2f, n=%d)",
        threshold, float(np.mean(genuine)), float(np.std(genuine)), len(genuine),
    )
    return round(threshold, 2)


def _per_class_metrics(
    confusion: np.ndarray, labels_sorted: List[int], label_map: Dict[int, str]
) -> List[ClassMetrics]:
    metrics: List[ClassMetrics] = []
    for i, label in enumerate(labels_sorted):
        true_positive = int(confusion[i, i])
        support = int(confusion[i].sum())
        predicted_positive = int(confusion[:, i].sum())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        metrics.append(
            ClassMetrics(
                student_id=label_map.get(label, str(label)),
                support=support,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
            )
        )
    return metrics


def sweep_thresholds(
    records: List[Tuple[int, int, float]], steps: int = 12
) -> List[Dict[str, float]]:
    """Accuracy / accept-rate as the distance threshold varies.

    ``records`` holds ``(true_label, predicted_label, distance)`` triples.
    """
    if not records:
        return []
    distances = [r[2] for r in records]
    low, high = min(distances), max(distances)
    if high <= low:
        high = low + 1.0
    sweep: List[Dict[str, float]] = []
    for threshold in np.linspace(low, high, steps):
        accepted = [r for r in records if r[2] <= threshold]
        correct_accepted = sum(1 for r in accepted if r[0] == r[1])
        wrong_accepted = len(accepted) - correct_accepted
        sweep.append(
            {
                "threshold": round(float(threshold), 2),
                "accept_rate": round(len(accepted) / len(records), 4),
                "accuracy": round(correct_accepted / len(records), 4),
                "false_accept_rate": round(wrong_accepted / len(records), 4),
            }
        )
    return sweep


# ----------------------------------------------------------------------
# Artefact writers
# ----------------------------------------------------------------------
def save_report(result: EvaluationResult, report_dir: str | Path) -> Dict[str, Path]:
    """Write metrics JSON, confusion-matrix CSV and (if possible) a PNG."""
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    json_path = directory / "evaluation.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    written["json"] = json_path

    csv_path = directory / "confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted"] + result.labels)
        for label, row in zip(result.labels, result.confusion):
            writer.writerow([label] + list(row))
    written["csv"] = csv_path

    png_path = _plot_confusion(result, directory)
    if png_path:
        written["png"] = png_path
    return written


def _plot_confusion(result: EvaluationResult, directory: Path) -> Optional[Path]:
    """Render the confusion matrix as a heat-map; skipped if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")                    # head-less backend
        import matplotlib.pyplot as plt
    except ImportError:                          # pragma: no cover
        logger.info("matplotlib not installed - skipping confusion-matrix plot")
        return None

    matrix = np.array(result.confusion)
    figure, axes = plt.subplots(figsize=(1.1 * len(result.labels) + 2,) * 2)
    image = axes.imshow(matrix, cmap="Blues")
    axes.set_xticks(range(len(result.labels)), result.labels, rotation=45, ha="right")
    axes.set_yticks(range(len(result.labels)), result.labels)
    axes.set_xlabel("Predicted")
    axes.set_ylabel("Actual")
    axes.set_title(
        f"Confusion matrix - {result.algorithm.upper()} "
        f"(accuracy {result.accuracy * 100:.1f}%)"
    )
    threshold_color = matrix.max() / 2 if matrix.size else 0
    for (row, column), value in np.ndenumerate(matrix):
        axes.text(
            column, row, str(value), ha="center", va="center",
            color="white" if value > threshold_color else "black", fontsize=9,
        )
    figure.colorbar(image, ax=axes, shrink=0.8)
    figure.tight_layout()
    path = directory / "confusion_matrix.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
