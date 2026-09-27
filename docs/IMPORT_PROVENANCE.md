# Import provenance

- Source: https://github.com/UWARG/autonomy-monorepo/tree/xfeat/airside/benchmarking
- Snapshot: `ca73594a1b87f37ffc05f8195fb388c1b2ef59c6`
- Imported all benchmarking files: compare.py, manifest, and 12 PNGs.
- Assets are unchanged. Original script is retained as `compare_upstream.py`.
- Adapted `compare.py` adds script-relative paths, output creation, missing-feature
  handling, CLI, XFeat, RANSAC diagnostics, and CSV output.
- Source MIT notice: `docs/UPSTREAM-LICENSE`.
- Submodule: https://github.com/verlab/accelerated_features.git
- Pinned to the original gitlink: `e92685f57f8318b18725c5c8c0bd28c7fe188d9a`.
  Its license and pretrained weights remain inside the submodule.
