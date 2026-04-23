from __future__ import annotations

"""Image preprocessing and visualization helpers for the rainfall CNN.

These functions convert uploaded images into 28x28 explainable tensors and also
prepare visual artifacts (feature maps, heatmaps, previews) for the frontend.
"""

import base64
import io
import math
from typing import Iterable

import numpy as np
from PIL import Image


def clamp01(array: np.ndarray) -> np.ndarray:
    """Clamp all numeric values to the [0, 1] range."""
    return np.clip(array, 0.0, 1.0)


def normalize_zero_one(array: np.ndarray) -> np.ndarray:
    """Min-max normalize an array into [0, 1], handling flat arrays safely."""
    array = array.astype(np.float64)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum - minimum < 1e-12:
        return np.zeros_like(array, dtype=np.float64)
    return (array - minimum) / (maximum - minimum)


def matrix_preview(array: np.ndarray, rows: int = 6, cols: int = 6) -> list[list[float]]:
    """Return a rounded small matrix snippet for UI/debug display."""
    array = np.asarray(array)
    if array.ndim == 1:
        preview = array[: rows * cols]
        return [[round(float(value), 4) for value in preview]]
    preview = array[:rows, :cols]
    return [[round(float(value), 4) for value in row] for row in preview]


def stats_for(array: np.ndarray) -> dict:
    """Return basic descriptive statistics for visualization cards."""
    array = np.asarray(array, dtype=np.float64)
    return {
        "shape": list(array.shape),
        "min": round(float(array.min()), 6),
        "max": round(float(array.max()), 6),
        "mean": round(float(array.mean()), 6),
        "std": round(float(array.std()), 6)
    }


def array_to_base64_png(array: np.ndarray, orientational: bool = False) -> str:
    """Encode a 2D numeric array as a grayscale PNG (base64 string)."""
    data = np.asarray(array, dtype=np.float64)
    if orientational:
        data = (data + math.pi) / (2.0 * math.pi)
    else:
        data = normalize_zero_one(data)

    image = Image.fromarray(np.uint8(clamp01(data) * 255), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def rgb_array_to_base64_png(array: np.ndarray) -> str:
    """Encode an RGB float array as a PNG base64 string."""
    data = np.asarray(array, dtype=np.float64)
    image = Image.fromarray(np.uint8(clamp01(data) * 255), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _interpolate_color(value: float, low: tuple[int, int, int], high: tuple[int,int, int]) -> tuple[int, int, int]:
    """Linearly blend between two RGB colors."""
    return tuple(
        int(round((1.0 - value) * low[channel] + value * high[channel])) for channel in range(3)
    )


def heatmap_to_rgb(array: np.ndarray) -> np.ndarray:
    """Map normalized heatmap values to a blue->green->yellow->red color scale."""
    data = normalize_zero_one(np.asarray(array, dtype=np.float64))
    height, width = data.shape
    rgb = np.zeros((height, width, 3), dtype=np.float64)

    anchors = [
        (0.0, (29, 78, 216)),
        (0.33, (34, 197, 94)),
        (0.66, (250, 204, 21)),
        (1.0, (239, 68, 68))
    ]

    for row in range(height):
        for col in range(width):
            value = float(data[row, col])
            for index in range(len(anchors) - 1):
                start_value, start_color = anchors[index]
                end_value, end_color = anchors[index + 1]
                if start_value <= value <= end_value:
                    blend = (value - start_value) / max(end_value - start_value, 1e-12)
                    rgb[row, col] = np.asarray(
                        _interpolate_color(blend, start_color, end_color),
                        dtype=np.float64
                    ) / 255.0
                    break

    return rgb


def heatmap_to_base64_png(array: np.ndarray) -> str:
    """Encode a heatmap into a frontend-friendly base64 PNG."""
    return rgb_array_to_base64_png(heatmap_to_rgb(array))


def overlay_heatmap_on_grayscale(grayscale: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap over grayscale input to visualize attention regions."""
    base = normalize_zero_one(np.asarray(grayscale, dtype=np.float64))
    base_rgb = np.stack([base, base, base], axis=-1)
    heatmap_rgb = heatmap_to_rgb(heatmap)
    return clamp01(((1.0 - alpha) * base_rgb) + (alpha * heatmap_rgb))


def image_bytes_to_rgb(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into an RGB float array in [0,1]."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.asarray(image, dtype=np.float64) / 255.0


def assess_cloud_image(file_bytes: bytes) -> dict:
    """Estimate whether the image looks atmospheric (cloud/sky) and return diagnostics."""
    rgb = image_bytes_to_rgb(file_bytes)
    grayscale = rgb_to_grayscale(rgb)

    max_channel = np.max(rgb, axis=2)
    min_channel = np.min(rgb, axis=2)
    saturation = np.divide(
        max_channel - min_channel,
        np.where(max_channel == 0, 1.0, max_channel)
    )

    bright_ratio = float(np.mean(grayscale > 0.48))
    low_sat_ratio = float(np.mean(saturation < 0.42))
    mid_sat_ratio = float(np.mean((saturation >= 0.18) & (saturation <= 0.75)))
    blue_sky_ratio = float(np.mean((rgb[:, :, 2] > rgb[:, :, 0]) & (rgb[:, :, 2] > rgb[:, :, 1]) & (rgb[:, :, 2] > 0.42)))
    contrast = float(np.std(grayscale))

    cloud_score = (
        (0.42 * bright_ratio)
        + (0.36 * low_sat_ratio)
        + (0.22 * min(1.0, blue_sky_ratio * 1.8))
    )
    sky_score = (
        (0.50 * min(1.0, blue_sky_ratio * 1.6))
        + (0.25 * bright_ratio)
        + (0.25 * mid_sat_ratio)
    )
    atmospheric_score = max(
        cloud_score,
        sky_score,
        (0.50 * bright_ratio) + (0.25 * low_sat_ratio) + (0.25 * mid_sat_ratio)
    )

    if contrast < 0.05 and bright_ratio < 0.12:
        cloud_score -= 0.08
        sky_score -= 0.05
        atmospheric_score -= 0.12

    cloud_score = float(max(0.0, min(1.0, cloud_score)))
    sky_score = float(max(0.0, min(1.0, sky_score)))
    atmospheric_score = float(max(0.0, min(1.0, atmospheric_score)))

    is_cloud_image = (
        cloud_score >= 0.36
        and bright_ratio >= 0.08
    )
    is_sky_image = (
        sky_score >= 0.30
        and bright_ratio >= 0.10
    )
    is_atmospheric_image = (
        atmospheric_score >= 0.24
        and (bright_ratio >= 0.06 or blue_sky_ratio >= 0.05 or low_sat_ratio >= 0.12)
    )

    if is_cloud_image:
        image_type = "Cloud-dominant atmosphere image"
    elif is_sky_image:
        image_type = "Sky-dominant atmosphere image"
    elif is_atmospheric_image:
        image_type = "General atmosphere image"
    else:
        image_type = "Non-atmospheric image"

    # Product requirement: do not block user uploads; provide diagnostics only.
    accepted = True

    return {
        "isCloudImage": bool(is_cloud_image),
        "isSkyImage": bool(is_sky_image),
        "isAtmosphericImage": bool(is_atmospheric_image),
        "acceptedForPrediction": accepted,
        "imageType": image_type,
        "cloudScore": round(cloud_score, 6),
        "skyScore": round(sky_score, 6),
        "atmosphericScore": round(atmospheric_score, 6),
        "metrics": {
            "brightPixelRatio": round(bright_ratio, 6),
            "lowSaturationRatio": round(low_sat_ratio, 6),
            "midSaturationRatio": round(mid_sat_ratio, 6),
            "blueSkyRatio": round(blue_sky_ratio, 6),
            "contrast": round(contrast, 6)
        },
        "message": (
            f"{image_type} accepted for rainfall prediction. "
            "Atmospheric confidence metrics are shown for interpretation only."
        )
    }

def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array to grayscale using standard luminance weights."""
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def resize_grayscale_image(grayscale: np.ndarray, size: int = 28) -> np.ndarray:
    """Resize grayscale array to model input size using bilinear interpolation."""
    image = Image.fromarray(np.uint8(clamp01(grayscale) * 255), mode="L")
    image = image.resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float64) / 255.0


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Build a normalized square Gaussian kernel."""
    radius = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)
    total = 0.0
    for row in range(size):
        for col in range(size):
            x = row - radius
            y = col - radius
            exponent = -((x * x + y * y) / (2.0 * sigma * sigma))
            value = math.exp(exponent)
            kernel[row, col] = value
            total += value
    return kernel / total


def convolve2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Run a manual 2D convolution with same-size output using edge padding."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    padding = kernel.shape[0] // 2
    padded = np.pad(image, ((padding, padding), (padding, padding)), mode="edge")
    output = np.zeros_like(image, dtype=np.float64)

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            patch = padded[row : row + kernel.shape[0], col : col + kernel.shape[1]]
            output[row, col] = float(np.sum(patch * kernel))

    return output


def gradient_magnitude_orientation(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute gradient magnitude and direction via finite differences."""
    height, width = image.shape
    padded = np.pad(image, ((1, 1), (1, 1)), mode="edge")
    magnitude = np.zeros_like(image, dtype=np.float64)
    orientation = np.zeros_like(image, dtype=np.float64)

    for row in range(height):
        for col in range(width):
            gx = padded[row + 1, col + 2] - padded[row + 1, col]
            gy = padded[row + 2, col + 1] - padded[row, col + 1]
            magnitude[row, col] = math.sqrt(float(gx * gx + gy * gy))
            orientation[row, col] = math.atan2(float(gy), float(gx))

    return magnitude, orientation


def preprocess_cloud_image(file_bytes: bytes, image_size: int = 28) -> dict:
    """Convert uploaded image bytes into the final explainable CNN input tensor."""
    rgb = image_bytes_to_rgb(file_bytes)
    grayscale = rgb_to_grayscale(rgb)
    resized = resize_grayscale_image(grayscale, image_size)

    kernel_small = gaussian_kernel(size=5, sigma=1.0)
    kernel_large = gaussian_kernel(size=7, sigma=1.6)

    blur_small = convolve2d_same(resized, kernel_small)
    blur_large = convolve2d_same(resized, kernel_large)
    dog = blur_small - blur_large
    magnitude, orientation = gradient_magnitude_orientation(blur_small)

    dog_norm = normalize_zero_one(dog)
    mag_norm = normalize_zero_one(magnitude)
    processed = clamp01((0.55 * resized) + (0.25 * dog_norm) + (0.20 * mag_norm))

    logs = [
        "Converted the uploaded cloud image to grayscale using 0.299R + 0.587G + 0.114B.",
        f"Resized the image to {image_size}x{image_size} so it matches the paper-sized CNN input.",
        "Computed two Gaussian blurs and subtracted them to form a Difference-of-Gaussians texture map.",
        "Estimated local edge strength and orientation with finite-difference gradients.",
        "Blended grayscale, DoG, and gradient magnitude into one explainable CNN input tensor."
    ]

    formula_notes = [
        {
          "name": "Grayscale",
          "formula": "I_gray = 0.299R + 0.587G + 0.114B",
          "array": resized,
          "description": "Brightness-only view of the uploaded cloud image."
        },
        {
          "name": "Gaussian Blur (sigma=1.0)",
          "formula": "G_sigma * I where G(x,y) = exp(-(x^2+y^2)/(2sigma^2)) / sum(G)",
          "array": blur_small,
          "description": "Smooth cloud structure for stable texture extraction."
        },
        {
          "name": "Gaussian Blur (sigma=1.6)",
          "formula": "G_sigma * I with a wider blur radius",
          "array": blur_large,
          "description": "Broader blur used for scale contrast."
        },
        {
          "name": "Difference of Gaussians",
          "formula": "DoG = (G_1.0 * I) - (G_1.6 * I)",
          "array": dog,
          "description": "Highlights cloud texture changes and blob-like structures."
        },
        {
          "name": "Gradient Magnitude",
          "formula": "m(x,y) = sqrt(gx^2 + gy^2)",
          "array": magnitude,
          "description": "Emphasizes edge strength in the cloud boundaries."
        },
        {
          "name": "Gradient Orientation",
          "formula": "theta(x,y) = atan2(gy, gx)",
          "array": orientation,
          "description": "Shows the dominant local direction of cloud edges.",
          "orientational": True
        },
        {
          "name": "Model Input",
          "formula": "X = clamp(0.55I_gray + 0.25norm(DoG) + 0.20norm(m), 0, 1)",
          "array": processed,
          "description": "Final single-channel tensor used by the CNN."
        }
    ]

    stages = []
    for note in formula_notes:
        array = note["array"]
        stages.append(
            {
                "name": note["name"],
                "description": note["description"],
                "formula": note["formula"],
                "stats": stats_for(array),
                "matrixPreview": matrix_preview(array),
                "imageBase64": array_to_base64_png(array, orientational=note.get("orientational", False))
            }
        )

    return {
        "processed": processed,
        "preprocessingStages": stages,
        "logs": logs,
        "originalPreviewBase64": array_to_base64_png(resized)
    }


def encode_feature_maps(name: str, arrays: Iterable[np.ndarray], formula: str, description: str) -> dict:
    """Package feature maps with previews/stats for frontend rendering."""
    maps = []
    for index, array in enumerate(arrays, start=1):
        maps.append(
            {
                "name": f"{name} Map {index}",
                "imageBase64": array_to_base64_png(array),
                "stats": stats_for(array),
                "matrixPreview": matrix_preview(array)
            }
        )

    return {
        "name": name,
        "formula": formula,
        "description": description,
        "maps": maps
    }

