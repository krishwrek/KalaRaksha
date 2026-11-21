"""
KalaRaksha - Preprocessing Module
Heritage Wall Deterioration Analysis & Restoration-Priority Mapping System

This module handles:
1. Dynamic input loading (path, bytes, PIL, numpy array)
2. Aspect-ratio preserving downsampling to max 1280px
3. BGR to RGB color normalization
4. LAB-space Contrast Limited Adaptive Histogram Equalization (CLAHE)
5. Non-linear Gamma Correction
6. Edge-preserving bilateral or median noise reduction
"""

from typing import Union, Tuple, Dict, Any
import numpy as np
import cv2
from PIL import Image
import io


def load_and_resize_image(
    image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray],
    max_dim: int = 1280
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Load an image from various sources, normalize to RGB, and resize so that
    max(height, width) <= max_dim while preserving the original aspect ratio.

    Returns:
        rgb_image: np.ndarray (H, W, 3) in uint8 RGB format.
        scale_factor: float ratio of new dimensions relative to original.
        orig_dims: Tuple of (original_height, original_width).
    """
    # 1. Load into numpy array
    if isinstance(image_input, str):
        # File path
        bgr = cv2.imread(image_input, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image from path: {image_input}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, (bytes, io.BytesIO)):
        if isinstance(image_input, io.BytesIO):
            data = image_input.getvalue()
        else:
            data = image_input
        np_arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image from byte buffer.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        # PIL Image
        rgb = np.array(image_input.convert("RGB"), dtype=np.uint8)
    elif isinstance(image_input, np.ndarray):
        # Check shape and channels
        if image_input.ndim == 2:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        elif image_input.shape[2] == 3:
            rgb = image_input.copy()
        else:
            raise ValueError(f"Unsupported image array shape: {image_input.shape}")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    orig_h, orig_w = rgb.shape[:2]
    max_side = max(orig_h, orig_w)

    if max_side > max_dim:
        scale_factor = max_dim / float(max_side)
        new_w = int(round(orig_w * scale_factor))
        new_h = int(round(orig_h * scale_factor))
        resized_rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale_factor = 1.0
        resized_rgb = rgb.copy()

    return resized_rgb, scale_factor, (orig_h, orig_w)


def apply_clahe_enhancement(
    rgb_image: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) in LAB space.
    Enhances local surface texture and micro-fractures without distorting natural stone hues.
    """
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return rgb_enhanced


def apply_gamma_correction(rgb_image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply non-linear gamma correction to lift deep shadows or prevent highlight clipping.
    Formula: I_out = 255 * (I_in / 255) ** (1 / gamma)
    """
    if abs(gamma - 1.0) < 1e-3 or gamma <= 0:
        return rgb_image

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(rgb_image, table)


def apply_noise_filtering(
    rgb_image: np.ndarray,
    filter_type: str = "bilateral",
    kernel_size: int = 5
) -> np.ndarray:
    """
    Apply noise filtering to suppress high-frequency sensor noise while preserving
    structural crack edges and flaking paint contours.
    Supported types: 'bilateral' (default, edge-preserving), 'median', 'gaussian', 'none'.
    """
    filter_lower = filter_type.lower()
    if filter_lower == "bilateral":
        # Bilateral filter operates on each channel or directly on color image
        return cv2.bilateralFilter(rgb_image, d=7, sigmaColor=40, sigmaSpace=40)
    elif filter_lower == "median":
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.medianBlur(rgb_image, k)
    elif filter_lower == "gaussian":
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.GaussianBlur(rgb_image, (k, k), 0)
    elif filter_lower == "none":
        return rgb_image
    else:
        # Fallback to bilateral
        return cv2.bilateralFilter(rgb_image, d=7, sigmaColor=40, sigmaSpace=40)


def preprocess_heritage_image(
    image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray],
    max_dim: int = 1280,
    apply_clahe: bool = True,
    clahe_clip: float = 2.5,
    gamma: float = 1.0,
    filter_type: str = "bilateral"
) -> Dict[str, Any]:
    """
    Complete end-to-end preprocessing pipeline for heritage wall inspection.

    Returns:
        dict containing:
            - 'original_rgb': unenhanced resized RGB image
            - 'enhanced_rgb': fully preprocessed RGB image
            - 'gray': single-channel grayscale uint8 of enhanced image
            - 'hsv': HSV representation of enhanced image
            - 'lab': LAB representation of enhanced image
            - 'scale_factor': resize scale factor
            - 'original_dimensions': (H, W) before resize
            - 'dimensions': (H, W) of processed image
            - 'metrics': brightness, contrast, and dynamic range metrics
    """
    resized_rgb, scale_factor, orig_dims = load_and_resize_image(image_input, max_dim=max_dim)

    # 1. Start with resized RGB
    current = resized_rgb.copy()

    # 2. Noise suppression first (preserves edges if bilateral)
    if filter_type != "none":
        current = apply_noise_filtering(current, filter_type=filter_type)

    # 3. CLAHE local contrast enhancement
    if apply_clahe:
        current = apply_clahe_enhancement(current, clip_limit=clahe_clip)

    # 4. Optional gamma correction
    if abs(gamma - 1.0) > 1e-3:
        current = apply_gamma_correction(current, gamma=gamma)

    # 5. Derived color spaces
    gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(current, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(current, cv2.COLOR_RGB2LAB)

    # Calculate basic image quality metrics
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    min_val, max_val = int(np.min(gray)), int(np.max(gray))

    return {
        "original_rgb": resized_rgb,
        "enhanced_rgb": current,
        "gray": gray,
        "hsv": hsv,
        "lab": lab,
        "scale_factor": scale_factor,
        "original_dimensions": orig_dims,
        "dimensions": gray.shape,
        "metrics": {
            "mean_luminance": round(mean_val, 2),
            "contrast_std": round(std_val, 2),
            "dynamic_range": [min_val, max_val],
            "filter_applied": filter_type,
            "clahe_applied": apply_clahe,
            "gamma_applied": gamma,
        }
    }
