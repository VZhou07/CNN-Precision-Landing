# CNN-Precision-Landing
Autonomous Precision Landing Driven by XFeat (a Lightweight Convolutional Neural Network)

Includes the benchmarking code and six image pairs from UWARG's `xfeat` branch,
the pinned `accelerated_features` submodule, and an offline landing scaffold.

From the repository root (Python 3.10+):

```powershell
git submodule update --init --recursive
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python airside/benchmarking/compare.py
```

For ORB versus XFeat:

```powershell
python -m pip install -r requirements-xfeat.txt
python airside/benchmarking/compare.py --method both --top-k 500
```

XFeat uses bundled weights and automatically selects CUDA when available.
Results go to `airside/benchmarking/results/`: match previews (up to 100 matches)
and `metrics.csv` (all matches). Use `--output` to keep separate runs; rerunning
in the same folder overwrites matching filenames. Default paths are relative
to the script. Inliers indicate geometric consistency, not ground-truth accuracy;
timings are preliminary single-pass measurements.

Complete [the landing scaffold](airside/landing/pipeline.py) using
[the remaining-work checklist](docs/NEXT_STEPS.md). See
[import provenance](docs/IMPORT_PROVENANCE.md) for commits and licensing.
`compare_upstream.py` preserves the original script for reference; run `compare.py`.
