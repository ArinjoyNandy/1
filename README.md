# Crack Particle Counter

This repository contains a small OpenCV-based utility to estimate how many particles
(segments/fragments) a brittle specimen has broken into after crack development.

## Files

- `count_particles.py` – CLI script that performs the image processing and counting.
- `RESULT.md` – initial estimate report for the provided sample image.

## How the script works

The counting pipeline has 3 main stages:

1. **Detect specimen region (`specimen_mask`)**
   - Convert image to grayscale.
   - Blur to reduce noise.
   - Apply Otsu thresholding.
   - Keep only the largest connected bright region.
   - Apply morphological closing to fill small holes.

2. **Detect cracks (`crack_mask`)**
   - Enhance contrast using CLAHE.
   - Use blackhat morphology to highlight dark crack-like ridges.
   - Threshold to binary crack mask.
   - Restrict crack mask to specimen area.
   - Dilate + close to connect tiny crack gaps.

3. **Count particles (`count_particles`)**
   - Remove cracks from specimen mask.
   - Clean residual speckles with morphological opening.
   - Use connected components analysis.
   - Filter out tiny components using a minimum-area threshold.
   - Count remaining components as particles.
   - Optionally label each particle index on an overlay image.

## Usage

```bash
python count_particles.py path/to/image.jpg
```

Optional arguments:

```bash
python count_particles.py path/to/image.jpg \
  --min-area 350 \
  --save-overlay overlay.png
```

- `--min-area`: ignore tiny regions (in pixels) that are likely noise.
- `--save-overlay`: write an image where detected particles are numbered for review.

## Example output

The script prints the estimated particle count to stdout:

```text
80
```

## Notes and limitations

- The method is an **estimate**, not a guaranteed exact count.
- Strong glare, low contrast cracks, and merged crack boundaries can affect detection.
- Best practice is to review the optional overlay image and tune `--min-area` per dataset.
- For production use, calibration with manually annotated images is recommended.
