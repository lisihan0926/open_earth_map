"""Render existing confusion CSVs without running model inference."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size, bold=False):
    filename = "arialbd.ttf" if bold else "arial.ttf"
    for candidate in (Path("C:/Windows/Fonts") / filename, Path(filename)):
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            pass
    return ImageFont.truetype("DejaVuSans.ttf", size)


def plot(csv_path, font_size=32):
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    names = rows[0][1:]
    matrix = np.array([[int(value) for value in row[1:]] for row in rows[1:]], dtype=np.int64)
    if matrix.shape != (len(names), len(names)) or (matrix < 0).any():
        raise ValueError(f"Invalid confusion matrix: {csv_path}")
    if [row[0] for row in rows[1:]] != names:
        raise ValueError("Row and column class names must match")
    support = matrix.sum(axis=1)
    values = np.divide(matrix, support[:, None], out=np.zeros_like(matrix, dtype=float), where=support[:, None] != 0)
    cell = max(150, font_size * 5)
    left, top = max(330, font_size * 11), 260
    side = len(names) * cell
    image = Image.new("RGB", (left + side + 210, top + side + 220), "white")
    draw = ImageDraw.Draw(image)
    label_font, value_font = font(font_size), font(font_size, True)
    title = csv_path.parents[2].name.replace("_", " ").title()
    draw.text((left + side // 2, 55), f"{title} | Confusion matrix", font=font(font_size + 16, True), fill="#172b4d", anchor="mm")
    draw.text((left + side // 2, 115), "Row-normalized pixel proportions", font=font(font_size - 4), fill="#526176", anchor="mm")
    for i, name in enumerate(names):
        y = top + i * cell
        draw.text((left - 24, y + cell // 2), name, font=label_font, fill="#25354a", anchor="rm")
        draw.text((left + i * cell + cell // 2, top - 30), name, font=font(font_size - 4), fill="#25354a", anchor="mb")
        for j in range(len(names)):
            value = values[i, j]
            color = tuple(int(a + (b-a)*value) for a, b in zip((243,248,253), (8,60,120)))
            x = left + j * cell
            draw.rectangle((x,y,x+cell,y+cell), fill=color, outline="white", width=3)
            text = f"{value*100:.1f}%" if support[i] else "N/A"
            draw.text((x+cell//2,y+cell//2), text, font=value_font, fill="white" if value > .52 else "#20344c", anchor="mm")
    draw.text((left + side//2, top+side+65), "Predicted class", font=font(font_size+4, True), fill="#172b4d", anchor="mm")
    draw.text((25, 195), "True class", font=font(font_size+4, True), fill="#172b4d")
    for y in range(side):
        value = 1-y/max(side-1,1)
        color = tuple(int(a+(b-a)*value) for a,b in zip((243,248,253),(8,60,120)))
        draw.line((left+side+45, top+y, left+side+85, top+y), fill=color)
    for value in (0, .25, .5, .75, 1):
        draw.text((left+side+100, top+int((1-value)*side)), f"{value:.0%}", font=font(font_size-6), fill="#25354a", anchor="lm")
    draw.text((left, top+side+135), "Unknown truth ignored; N/A = no labeled pixels in that row.", font=font(font_size-6), fill="#526176")
    destination = csv_path.with_suffix(".png")
    image.save(destination, dpi=(300,300))
    print("Saved:", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", type=Path, help="CSV path; omitted: render all experiment matrices")
    parser.add_argument("--font-size", type=int, default=32)
    args = parser.parse_args()
    if args.font_size < 16:
        parser.error("--font-size must be at least 16")
    paths = [args.csv] if args.csv else sorted((Path(__file__).resolve().parent / "outputs_comparison").glob("*/seed_*/predictions/confusion_matrix.csv"))
    if not paths:
        parser.error("No confusion_matrix.csv found. Run evaluation first.")
    for path in paths:
        plot(path, args.font_size)
