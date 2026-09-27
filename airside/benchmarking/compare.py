"""Run ORB / XFeat on the imported teach-repeat pairs."""
import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def orb_matches(left, right, top_k=500, ratio=0.55):
    orb = cv2.ORB_create(nfeatures=top_k)
    kp1, des1 = orb.detectAndCompute(left, None)
    kp2, des2 = orb.detectAndCompute(right, None)
    matches = []
    if des1 is not None and des2 is not None and len(des2) >= 2:
        candidates = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
        matches = [m for pair in candidates if len(pair) == 2
                   for m, n in [pair] if m.distance < ratio * n.distance]
        matches.sort(key=lambda m: m.distance)
    return (np.array([kp1[m.queryIdx].pt for m in matches], np.float32).reshape(-1, 2),
            np.array([kp2[m.trainIdx].pt for m in matches], np.float32).reshape(-1, 2))


def load_xfeat(top_k):
    submodule = HERE.parents[1] / "accelerated_features"
    if not (submodule / "modules/xfeat.py").is_file():
        raise RuntimeError("Run: git submodule update --init --recursive")
    sys.path.insert(0, str(submodule))
    from modules.xfeat import XFeat
    return XFeat(top_k=top_k)


def geometry(points1, points2):
    """RANSAC consistency is not a ground-truth accuracy measurement."""
    if len(points1) < 4:
        return None, np.zeros(len(points1), dtype=bool)
    transform, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 3.0)
    if transform is None or mask is None:
        return None, np.zeros(len(points1), dtype=bool)
    return transform, mask.ravel().astype(bool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("orb", "xfeat", "both"), default="orb")
    parser.add_argument("--assets", type=Path, default=HERE / "assets")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--ratio", type=float, default=0.55, help="ORB Lowe ratio")
    args = parser.parse_args()
    if args.top_k <= 0 or not 0 < args.ratio < 1:
        parser.error("--top-k must be positive and --ratio must be between 0 and 1")
    pairs = json.loads((args.assets / "manifest.json").read_text(encoding="utf-8"))
    if not pairs:
        parser.error("The manifest is empty")
    methods = ("orb", "xfeat") if args.method == "both" else (args.method,)
    model = load_xfeat(args.top_k) if "xfeat" in methods else None
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in pairs:
        images = [cv2.imread(str(args.assets / pair[key]), cv2.IMREAD_GRAYSCALE)
                  for key in ("left", "right")]
        if any(image is None for image in images):
            raise FileNotFoundError(f"Cannot read pair {pair['pair']} in {args.assets}")
        for method in methods:
            started = perf_counter()
            if method == "orb":
                points1, points2 = orb_matches(*images, args.top_k, args.ratio)
                device = "cpu"
            else:
                # Upstream numpy parser expects HWC uint8 and normalizes to [0, 1].
                points1, points2 = model.match_xfeat(images[0][..., None], images[1][..., None])
                device = str(model.dev)
            elapsed_ms = (perf_counter() - started) * 1000
            _, inliers = geometry(points1, points2)
            row = dict(pair=pair["pair"], method=method, device=device, top_k=args.top_k,
                       matches=len(points1), inliers=int(inliers.sum()),
                       inlier_ratio=float(inliers.mean()) if len(inliers) else 0.0,
                       matching_ms=round(elapsed_ms, 3))
            rows.append(row)
            print(row)
            kp1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in points1[:100]]
            kp2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in points2[:100]]
            matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]
            preview = cv2.drawMatches(images[0], kp1, images[1], kp2, matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            output = args.output / f"{method}_pair{pair['pair']}.png"
            if not cv2.imwrite(str(output), preview):
                raise OSError(f"Cannot write {output}")
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
