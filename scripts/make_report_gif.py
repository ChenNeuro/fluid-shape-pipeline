from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_FRAMES = [
    ("separability_pca.png", "Feature Separability (PCA)"),
    ("spectra_examples.png", "Representative Spectra"),
    ("confusion_matrix.png", "Baseline Confusion Matrix"),
    ("robustness_sweep.png", "Baseline Robustness Sweep"),
    ("holdout_stability.png", "Multi-Seed Holdout Stability"),
    ("confusion_matrix_sota.png", "SOTA Candidate Confusion Matrix"),
    ("robustness_sweep_sota.png", "SOTA Candidate Robustness Sweep"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an animated GIF from report PNGs")
    parser.add_argument("--reports-dir", default="reports", help="Directory containing report images")
    parser.add_argument("--output", default="reports/pipeline_overview.gif", help="Output GIF path")
    parser.add_argument("--width", type=int, default=1000, help="GIF frame width")
    parser.add_argument("--height", type=int, default=620, help="GIF frame height")
    parser.add_argument("--duration-ms", type=int, default=1300, help="Frame duration in milliseconds")
    return parser.parse_args()


def make_frame(image_path: Path, caption: str, size: tuple[int, int]) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", (width, height), (245, 247, 250))

    title_h = 50
    inner_margin = 20

    image = Image.open(image_path).convert("RGB")
    fitted = ImageOps.contain(image, (width - 2 * inner_margin, height - title_h - 2 * inner_margin))

    x0 = (width - fitted.width) // 2
    y0 = title_h + (height - title_h - fitted.height) // 2
    canvas.paste(fitted, (x0, y0))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, title_h), fill=(15, 23, 42))

    font = ImageFont.load_default()
    draw.text((14, 16), caption, fill=(255, 255, 255), font=font)
    draw.text((width - 210, 16), image_path.name, fill=(203, 213, 225), font=font)

    return canvas


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    output_path = Path(args.output)

    frames = []
    used = []
    for file_name, caption in DEFAULT_FRAMES:
        image_path = reports_dir / file_name
        if not image_path.exists():
            continue
        frames.append(make_frame(image_path=image_path, caption=caption, size=(args.width, args.height)))
        used.append(file_name)

    if not frames:
        raise FileNotFoundError(f"No expected PNG frames found in {reports_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )

    print(f"GIF written to: {output_path}")
    print(f"Frames used ({len(used)}): {', '.join(used)}")


if __name__ == "__main__":
    main()
