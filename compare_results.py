"""Compare evaluated checkpoints only when test files and resolution match."""
import argparse
import csv
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "outputs_comparison"
    rows = []
    reference = None
    for variant in ("baseline", "quantum", "quantum_only"):
        path = root / variant / f"seed_{args.seed}" / "predictions" / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Run testing for {variant} first: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        signature = (data["test_files"], data["input_config"], data["evaluation_resolution"], data["device"])
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError("Test files, input configuration, evaluation resolution or device differ; rerun consistently")
        rows.append({"variant": variant, **{key: data[key] for key in ("mIoU", "OA", "macro_F1", "parameters", "seconds_per_image", "images")}})
    destination = root / f"comparison_seed_{args.seed}.csv"
    with destination.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print("Saved:", destination)
