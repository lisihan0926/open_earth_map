"""Pixel metrics. Rows are truth; columns are predictions; truth 0 is ignored."""
import csv
import json
import numpy as np


def confusion_counts(target, prediction, n_classes=9):
    if target.shape != prediction.shape:
        raise ValueError("Prediction and label shapes differ")
    valid = target != 0
    truth, pred = target[valid].astype(int), prediction[valid].astype(int)
    if ((truth < 0) | (truth >= n_classes) | (pred < 0) | (pred >= n_classes)).any():
        raise ValueError("Class IDs outside configured range")
    return np.bincount(truth * n_classes + pred, minlength=n_classes**2).reshape(n_classes, n_classes)


def export_metrics(matrix, output_dir, names, metadata):
    tp = np.diag(matrix).astype(float)
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    def ratio(a, b):
        return np.divide(a, b, out=np.full(a.shape, np.nan), where=b != 0)
    iou = ratio(tp, support + predicted - tp)
    precision = ratio(tp, predicted)
    recall = ratio(tp, support)
    f1 = ratio(2 * tp, support + predicted)
    def mean(values):
        values = values[1:]
        return float(np.nanmean(values)) if np.isfinite(values).any() else None
    def number(value):
        return float(value) if np.isfinite(value) else None
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true / predicted"] + names)
        writer.writerows([[name] + row.tolist() for name, row in zip(names, matrix)])
    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class", "support", "predicted", "iou", "precision", "recall", "f1"])
        for i in range(1, len(names)):
            writer.writerow([i, names[i], support[i], predicted[i], *[number(v[i]) for v in (iou, precision, recall, f1)]])
    summary = dict(metadata, valid_pixels=int(matrix.sum()), mIoU=mean(iou), macro_F1=mean(f1),
                   OA=float(tp.sum() / matrix.sum()) if matrix.sum() else None,
                   convention="Truth 0 ignored; predictions of 0 on valid truth count as errors. Undefined metrics are null; absent union classes excluded from mIoU.")
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
