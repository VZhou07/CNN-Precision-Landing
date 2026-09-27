"""Offline landing estimation scaffold. No flight commands are sent."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LandingEstimate:
    lateral_error_m: tuple[float, float]
    yaw_error_rad: float
    confidence: float


def preprocess_frame(frame: np.ndarray, camera_matrix: np.ndarray,
                     distortion: np.ndarray) -> np.ndarray:
    """Return a calibrated image in the same coordinates as the reference."""
    # TODO: measured calibration, undistortion, consistent resize/crop policy.
    raise NotImplementedError("Implement camera calibration and preprocessing")


def estimate_landing_error(reference_points: np.ndarray, live_points: np.ndarray,
                           camera_matrix: np.ndarray, altitude_m: float) -> LandingEstimate:
    """Convert verified correspondences to errors in a documented body frame."""
    # TODO: reject insufficient/degenerate matches, estimate geometry, resolve scale
    # from altitude, and document coordinate signs and yaw conventions.
    raise NotImplementedError("Implement calibrated geometry and confidence gates")


def process_frame(reference: np.ndarray, live_frame: np.ndarray,
                  camera_matrix: np.ndarray, distortion: np.ndarray,
                  altitude_m: float) -> LandingEstimate | None:
    """Return None for rejected observations; connect recorded frames first."""
    # TODO: preprocess, cache reference descriptors, match live features, estimate
    # error, and reject stale frames, outliers, and temporal jumps.
    raise NotImplementedError("Connect preprocessing, XFeat, and the estimator")
