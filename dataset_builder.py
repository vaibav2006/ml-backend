from __future__ import annotations

"""Dataset utilities for training the manual CNN.

The trainer can use:
1) a real labeled dataset from disk, or
2) a synthetic fallback dataset generated from cloud-like patterns.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from image_utils import preprocess_cloud_image


LABELS = ["no-rain", "rain"]


def ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def gaussian_blob(canvas: np.ndarray, center_x: float, center_y: float, sigma: float, amplitude: float) -> None:
    """Add one Gaussian cloud blob to a grayscale canvas in-place."""
    height, width = canvas.shape
    for row in range(height):
        for col in range(width):
            dx = row - center_x
            dy = col - center_y
            canvas[row, col] += amplitude * np.exp(-((dx * dx + dy * dy) / (2.0 * sigma * sigma)))


def vertical_rain_streaks(canvas: np.ndarray, rng: np.random.Generator, intensity: float) -> None:
    """Draw streak-like vertical rain traces used for rainy synthetic samples."""
    height, width = canvas.shape
    streak_count = int(width * 0.18)
    for _ in range(streak_count):
        col = int(rng.integers(0, width))
        start_row = int(rng.integers(0, height // 3))
        length = int(rng.integers(height // 4, height // 2))
        for row in range(start_row, min(height, start_row + length)):
            canvas[row, col] += intensity * (1.0 - ((row - start_row) / max(length, 1)))


def synthesize_cloud_image(label: int, seed: int, size: int = 72) -> np.ndarray:
    """Create a synthetic RGB cloud image for `no-rain` (0) or `rain` (1)."""
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size), dtype=np.float64)

    sky_start = 0.55 if label == 0 else 0.35
    sky_end = 0.82 if label == 0 else 0.55
    for row in range(size):
        blend = row / max(size - 1, 1)
        image[row, :] = sky_start * (1.0 - blend) + sky_end * blend

    blob_count = int(rng.integers(4, 7) if label == 0 else rng.integers(7, 12))
    for _ in range(blob_count):
        center_x = float(rng.uniform(12, size - 12))
        center_y = float(rng.uniform(12, size - 12))
        sigma = float(rng.uniform(5.0, 12.0) if label == 0 else rng.uniform(7.0, 15.0))
        amplitude = float(rng.uniform(0.12, 0.22) if label == 0 else rng.uniform(0.20, 0.35))
        gaussian_blob(image, center_x, center_y, sigma, amplitude)

    texture_noise = rng.normal(0.0, 0.025 if label == 0 else 0.05, size=(size, size))
    image += texture_noise

    if label == 1:
        image -= 0.18
        vertical_rain_streaks(image, rng, intensity=0.12)
        image += rng.normal(0.0, 0.03, size=(size, size))
    else:
        image += 0.05

    image = np.clip(image, 0.0, 1.0)
    rgb = np.stack([image, image, image], axis=-1)
    return rgb


def rgb_array_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode an RGB float array ([0,1]) into PNG bytes."""
    pil_image = Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255), mode="RGB")
    from io import BytesIO

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def save_preview_dataset(generated_root: Path, per_class: int = 4) -> None:
    """Save a few preview images per class so users can inspect generated data."""
    ensure_dir(generated_root)
    for label_index, label_name in enumerate(LABELS):
        label_dir = generated_root / label_name
        ensure_dir(label_dir)
        for image_index in range(per_class):
            rgb = synthesize_cloud_image(label_index, seed=(label_index * 1000) + image_index)
            path = label_dir / f"sample_{image_index + 1}.png"
            Image.fromarray(np.uint8(rgb * 255), mode="RGB").save(path)


def load_real_dataset(dataset_root: Path, image_size: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and preprocess labeled cloud images from `dataset_root/<label>` folders."""
    images = []
    labels = []
    logs = [f"Loading real cloud images from {dataset_root}."]

    for label_index, label_name in enumerate(LABELS):
        label_dir = dataset_root / label_name
        if not label_dir.exists():
            raise FileNotFoundError(f"Expected folder '{label_dir}' for label '{label_name}'.")
        for file_path in sorted(label_dir.iterdir()):
            if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            payload = preprocess_cloud_image(file_path.read_bytes(), image_size=image_size)
            images.append(payload["processed"])
            labels.append(label_index)

    logs.append(f"Loaded {len(images)} total images from the real dataset.")
    return np.asarray(images, dtype=np.float64), np.asarray(labels, dtype=np.int64), logs


def build_synthetic_dataset(image_size: int, samples_per_class: int, generated_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic cloud data and preprocess it into model-ready tensors."""
    ensure_dir(generated_root)
    save_preview_dataset(generated_root)

    images = []
    labels = []
    logs = [
        "No real dataset folder was supplied, so the trainer generated a synthetic cloud dataset.",
        "Synthetic rainy clouds are darker, denser, and include streak-like rain texture.",
        "Synthetic non-rain clouds are brighter, softer, and have weaker gradients."
    ]

    for label_index, label_name in enumerate(LABELS):
        label_dir = generated_root / label_name
        ensure_dir(label_dir)
        for image_index in range(samples_per_class):
            rgb = synthesize_cloud_image(label_index, seed=(label_index * 5000) + image_index + 17)
            png_bytes = rgb_array_to_png_bytes(rgb)
            payload = preprocess_cloud_image(png_bytes, image_size=image_size)
            images.append(payload["processed"])
            labels.append(label_index)

            if image_index < 4:
                preview_path = label_dir / f"generated_{image_index + 1}.png"
                Image.fromarray(np.uint8(rgb * 255), mode="RGB").save(preview_path)

    logs.append(f"Generated {samples_per_class * len(LABELS)} synthetic training examples.")
    return np.asarray(images, dtype=np.float64), np.asarray(labels, dtype=np.int64), logs


def split_dataset(images: np.ndarray, labels: np.ndarray, train_ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split arrays into train/validation partitions."""
    indices = np.arange(len(images))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    split_index = int(len(indices) * train_ratio)
    train_idx = indices[:split_index]
    val_idx = indices[split_index:]

    return images[train_idx], labels[train_idx], images[val_idx], labels[val_idx]


def load_or_generate_dataset(
    dataset_root: str | None,
    generated_root: str,
    image_size: int = 28,
    samples_per_class: int = 48
) -> dict:
    """Main entry point used by training: real dataset if available, else synthetic."""
    root = Path(generated_root)
    logs = []

    if dataset_root:
        dataset_path = Path(dataset_root)
        if dataset_path.exists():
            images, labels, local_logs = load_real_dataset(dataset_path, image_size=image_size)
            logs.extend(local_logs)
        else:
            images, labels, local_logs = build_synthetic_dataset(image_size, samples_per_class, root)
            logs.extend(
                [
                    f"Dataset path '{dataset_root}' was not found.",
                    "Falling back to the synthetic dataset so the demo remains runnable."
                ]
            )
            logs.extend(local_logs)
    else:
        images, labels, local_logs = build_synthetic_dataset(image_size, samples_per_class, root)
        logs.extend(local_logs)

    train_x, train_y, val_x, val_y = split_dataset(images, labels)

    return {
        "trainX": train_x[:, np.newaxis, :, :],
        "trainY": train_y,
        "valX": val_x[:, np.newaxis, :, :],
        "valY": val_y,
        "logs": logs,
        "source": "real-dataset" if dataset_root and Path(dataset_root).exists() else "synthetic-dataset"
    }
