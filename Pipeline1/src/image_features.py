"""Global grayscale and edge features for the Decision Tree experiments."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


MAX_IMAGE_SIZE = 512
EDGE_THRESHOLD = 0.06
ORIENTATION_BINS = np.linspace(0.0, 180.0, 7)
FEATURE_NAMES = (
    "edge_density",
    "horizontal_edge_strength",
    "vertical_edge_strength",
    "orientation_000_030",
    "orientation_030_060",
    "orientation_060_090",
    "orientation_090_120",
    "orientation_120_150",
    "orientation_150_180",
    "grayscale_contrast",
)


def load_grayscale(path: str | Path, max_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
    """Orient, downsize without cropping/upscaling, and smooth a grayscale image.

    A fixed one-pixel Gaussian blur reduces fine noise before edge extraction.
    The returned float32 intensities lie in [0, 1]. Source files are never written.
    """
    if max_size < 3:
        raise ValueError("max_size must be at least 3 pixels.")
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
        grayscale = np.asarray(image, dtype=np.float32) / 255.0
    if min(grayscale.shape) < 3:
        raise ValueError("Images must be at least 3 pixels wide and high.")
    return grayscale


def gradient_maps(grayscale: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized Sobel x/y gradients and their magnitude.

    Dividing Sobel kernels by 8 gives derivatives in normalized intensity units
    per pixel. Reflection padding avoids introducing an artificial black border.
    """
    gray = np.asarray(grayscale, dtype=np.float32)
    if gray.ndim != 2 or min(gray.shape) < 3:
        raise ValueError("Expected a two-dimensional grayscale image of size >= 3x3.")
    if not np.isfinite(gray).all() or gray.min() < 0 or gray.max() > 1:
        raise ValueError("Grayscale values must be finite and lie in [0, 1].")
    padded = np.pad(gray, 1, mode="reflect")
    gx = (
        padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:]
        - padded[:-2, :-2] - 2 * padded[1:-1, :-2] - padded[2:, :-2]
    ) / 8.0
    gy = (
        padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
        - padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:]
    ) / 8.0
    return gx, gy, np.hypot(gx, gy)


def features_from_grayscale(
    grayscale: np.ndarray, edge_threshold: float = EDGE_THRESHOLD
) -> dict[str, float]:
    """Summarize edges globally; no image regions or raw-pixel vectors are used.

    Orientation is the unsigned GRADIENT direction, normal to the edge:
    0 degrees responds to vertical edges and 90 degrees to horizontal edges.
    Histogram bins are left-closed, right-open, and weighted by edge magnitude.
    Only pixels above the fixed threshold enter the histogram. If none qualify,
    all orientation fractions are zero. All other means cover the full image.
    """
    if not np.isfinite(edge_threshold) or not 0 < edge_threshold < 1:
        raise ValueError("edge_threshold must be finite and lie strictly between 0 and 1.")
    gx, gy, magnitude = gradient_maps(grayscale)
    edges = magnitude > edge_threshold
    angles = np.degrees(np.arctan2(gy, gx)).astype(np.float64) % 180.0
    histogram, _ = np.histogram(
        angles[edges], bins=ORIENTATION_BINS, weights=magnitude[edges].astype(np.float64)
    )
    if histogram.sum() > 0:
        histogram /= histogram.sum()
    values = [
        float(edges.mean()),
        float(np.abs(gy).mean()),
        float(np.abs(gx).mean()),
        *histogram.tolist(),
        float(np.asarray(grayscale).std()),
    ]
    return dict(zip(FEATURE_NAMES, values, strict=True))


def extract_image_features(
    path: str | Path,
    *,
    max_size: int = MAX_IMAGE_SIZE,
    edge_threshold: float = EDGE_THRESHOLD,
) -> dict[str, float]:
    """Extract the same ten features from either an original or augmented image."""
    grayscale = load_grayscale(path, max_size=max_size)
    return features_from_grayscale(grayscale, edge_threshold=edge_threshold)
