#!/usr/bin/env python3
"""Estimate crack-separated particle count in a brittle-fracture image.

Usage:
  python count_particles.py path/to/image.jpg

The script detects crack lines as dark ridges, closes gaps, and counts connected
regions inside the specimen mask.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def specimen_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Keep the largest connected bright area (glass specimen + highlights).
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th)
    if n_labels <= 1:
        return th
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Fill small holes.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def crack_mask(image: np.ndarray, specimen: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Crack lines are generally darker than neighboring regions.
    blackhat = cv2.morphologyEx(
        clahe,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )

    _, cracks = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cracks = cv2.bitwise_and(cracks, specimen)

    # Thicken and bridge tiny breaks in crack boundaries.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cracks = cv2.dilate(cracks, k, iterations=1)
    cracks = cv2.morphologyEx(cracks, cv2.MORPH_CLOSE, k, iterations=2)
    return cracks


def count_particles(image: np.ndarray, min_area: int) -> tuple[int, np.ndarray]:
    specimen = specimen_mask(image)
    cracks = crack_mask(image, specimen)

    # Particles are specimen regions left after removing cracks.
    particles = cv2.bitwise_and(specimen, cv2.bitwise_not(cracks))
    particles = cv2.morphologyEx(
        particles,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(particles)
    count = 0
    overlay = image.copy()

    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        count += 1
        ys, xs = np.where(labels == label)
        cy, cx = int(ys.mean()), int(xs.mean())
        cv2.putText(overlay, str(count), (cx - 8, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    return count, overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Count crack-separated particles")
    parser.add_argument("image", type=Path)
    parser.add_argument("--min-area", type=int, default=350, help="ignore tiny regions (pixels)")
    parser.add_argument("--save-overlay", type=Path, default=None, help="optional output image with labels")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    count, overlay = count_particles(image, args.min_area)
    print(count)

    if args.save_overlay:
        cv2.imwrite(str(args.save_overlay), overlay)


if __name__ == "__main__":
    main()
