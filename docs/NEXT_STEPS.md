# Remaining work

The source contained an ORB baseline on six image pairs. The adapted runner adds
XFeat, CSV metrics, and RANSAC diagnostics. Landing hooks intentionally raise
`NotImplementedError` until implemented.

- [ ] Review match previews and record failures for each manifest pair.
- [ ] Annotate ground-truth correspondences/transforms; RANSAC inliers alone do
  not establish correct matching or landing accuracy.
- [ ] Expand the dataset across altitude, lighting, viewpoint, and surface texture;
  separate tuning/evaluation sets. Supplied images contain overlays: obtain clean
  original frames or mask overlays before drawing conclusions.
- [ ] Add warm-up, repeated trials, median/p95 latency, hardware/version metadata,
  and CUDA synchronization if timing GPU operations separately. Current timing
  includes extraction, matching, and return to CPU, excludes model startup,
  geometry, disk I/O and drawing, and includes first-call overhead.
- [ ] Tune feature budgets and match filters. Both methods default to 500 features
  but use different matching rules.
- [ ] Implement `preprocess_frame` in `airside/landing/pipeline.py`: measured
  camera calibration, distortion correction, and teach/reference selection.
- [ ] Implement `estimate_landing_error`: validate the planar assumption, scale
  from altitude, camera/body transform, yaw conventions, and rejection gates.
- [ ] Implement `process_frame`: cached reference features, recorded video,
  timestamp checks, temporal consistency. Test known offsets and no-match cases.
- [ ] Define controller limits, invalid/stale-estimate behavior, and loss-of-target
  handling; validate offline and in simulation before flight integration.
- [ ] Add camera/telemetry adapters and measure latency on target hardware.
  No vehicle transport or flight commands are implemented here.
