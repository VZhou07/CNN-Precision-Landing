"""Run ORB / XFeat on the imported teach-repeat pairs."""
import argparse
import csv
import json
import platform
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


def load_xfeat(top_k, device="auto"):
    submodule = HERE.parents[1] / "accelerated_features"
    if not (submodule / "modules/xfeat.py").is_file():
        raise RuntimeError("Run: git submodule update --init --recursive")
    sys.path.insert(0, str(submodule))
    print("Loading PyTorch and XFeat (first import may take a few seconds)...", flush=True)
    import torch
    from modules.xfeat import XFeat
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but this PyTorch environment has no available CUDA device. Use --device cpu or auto.")
    model = XFeat(top_k=top_k)
    if device != "auto":
        model.dev = torch.device(device)
        model.to(model.dev)
    return model


def xfeat_matches(model, left, right, min_cossim=-1):
    """Upstream sparse XFeat + mutual nearest neighbours, with empty-image handling."""
    import torch
    with torch.inference_mode():
        features = [model.detectAndCompute(model.parse_input(image[..., None]))[0]
                    for image in (left, right)]
        if any(len(feature["keypoints"]) == 0 for feature in features):
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        indices = model.match(features[0]["descriptors"], features[1]["descriptors"],
                              min_cossim=min_cossim)
        return tuple(feature["keypoints"][index].cpu().numpy()
                     for feature, index in zip(features, indices))


def synchronize(model):
    if model is not None and model.dev.type == "cuda":
        import torch
        torch.cuda.synchronize(model.dev)


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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="XFeat device; ORB always uses CPU")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--min-cossim", type=float, default=-1,
                        help="XFeat cosine cutoff; -1 disables filtering, as upstream match_xfeat does")
    args = parser.parse_args()
    if args.top_k <= 0 or not 0 < args.ratio < 1:
        parser.error("--top-k must be positive and --ratio must be between 0 and 1")
    if args.warmup < 0 or args.repeats < 1 or not -1 <= args.min_cossim <= 1:
        parser.error("--warmup >= 0, --repeats >= 1, and -1 <= --min-cossim <= 1 required")
    pairs = json.loads((args.assets / "manifest.json").read_text(encoding="utf-8"))
    if not pairs:
        parser.error("The manifest is empty")
    methods = ("orb", "xfeat") if args.method == "both" else (args.method,)
    model = load_xfeat(args.top_k, args.device) if "xfeat" in methods else None
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    samples = []
    for pair in pairs:
        images = [cv2.imread(str(args.assets / pair[key]), cv2.IMREAD_GRAYSCALE)
                  for key in ("left", "right")]
        if any(image is None for image in images):
            raise FileNotFoundError(f"Cannot read pair {pair['pair']} in {args.assets}")
        for method in methods:
            if method == "orb":
                run = lambda: orb_matches(*images, args.top_k, args.ratio)
                device = "cpu"
            else:
                run = lambda: xfeat_matches(model, *images, args.min_cossim)
                device = str(model.dev)
            active_model = model if method == "xfeat" else None
            for _ in range(args.warmup):
                run()
            timings = []
            for repeat in range(args.repeats):
                synchronize(active_model)
                started = perf_counter()
                points1, points2 = run()
                synchronize(active_model)
                elapsed_ms = (perf_counter() - started) * 1000
                timings.append(elapsed_ms)
                samples.append(dict(pair=pair["pair"], method=method, repeat=repeat,
                                    matching_ms=elapsed_ms))
            cv2.setRNGSeed(0)
            _, inliers = geometry(points1, points2)
            row = dict(pair=pair["pair"], method=method, device=device, top_k=args.top_k,
                       matches=len(points1), inliers=int(inliers.sum()),
                       inlier_ratio=float(inliers.mean()) if len(inliers) else 0.0,
                       matching_ms_median=round(float(np.median(timings)), 3),
                       matching_ms_p95=round(float(np.percentile(timings, 95)), 3))
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
    with (args.output / "timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    metadata = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                    processor=platform.processor(), numpy=np.__version__, opencv=cv2.__version__,
                    opencv_threads=cv2.getNumThreads(),
                    arguments={key: str(value) if isinstance(value, Path) else value
                               for key, value in vars(args).items()},
                    timing_scope="Both image extractions and matching, including input/output transfers; excludes model loading, image reading, RANSAC, drawing and writing. Quality metrics use the final repeat.")
    if model is not None:
        import torch
        metadata.update(torch=torch.__version__, torch_threads=torch.get_num_threads(),
                        cuda_build=torch.version.cuda, xfeat_device=str(model.dev),
                        gpu=torch.cuda.get_device_name(model.dev) if model.dev.type == "cuda" else None)
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metrics, raw timings, environment metadata, and previews to {args.output}")


if __name__ == "__main__":
    main()
