import cv2
import numpy as np
from typing import Dict, Any, Tuple
import math
from sklearn.cluster import KMeans

ASSUMED_HORIZONTAL_FOV_DEGREES = 60.0
MAX_PROCESS_DIMENSION = 1024


def _downscale_if_needed(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Downscale very large images to speed up processing and return the scale used."""
    height, width = image.shape[:2]
    max_dim = max(height, width)
    if max_dim <= MAX_PROCESS_DIMENSION:
        return image, 1.0
    scale = MAX_PROCESS_DIMENSION / float(max_dim)
    new_size = (int(width * scale), int(height * scale))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def _maybe_enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE when contrast is low to help subsequent thresholding."""
    if gray.std() >= 15:
        return gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _adaptive_block_size(shape: Tuple[int, int]) -> int:
    min_dim = min(shape)
    candidate = max(3, min_dim // 8)
    if candidate % 2 == 0:
        candidate += 1
    return candidate


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1
    final_mask = np.zeros_like(mask)
    if areas[largest_idx - 1] > 0:
        final_mask[labels == largest_idx] = 255
    return final_mask


from typing import Optional, Tuple


def _detect_coin_and_get_scale(
    image: np.ndarray,
) -> Tuple[Optional[float], Optional[Tuple[int, int, int]], Optional[float]]:
    """
    Detects a coin (assuming US Penny ~1.905 cm diameter) in the image and returns pixels_per_cm,
    coin location, and tilt correction factor.

    Handles both circular and elliptical coins (when coin is at an angle).

    Returns:
        pixels_per_cm: Pixel-to-cm conversion factor
        coin_data: (x, y, radius) tuple
        tilt_correction: cos(θ) = minor_axis/major_axis for ellipses, 1.0 for circles
    """

    full_h, full_w = image.shape[:2]

    # Apply illumination correction first for better coin detection
    corrected_L = correct_illumination(image)

    # 1. Detect circles on illumination-corrected image
    min_radius_px = int(min(full_h, full_w) * 0.1)  # Reduced from 0.1 (smaller coins)
    max_radius_px = int(min(full_h, full_w) * 0.9)  # Slightly larger allowance for full coin detection

    gray_blurred = cv2.GaussianBlur(corrected_L, (9, 9), 2)

    min_radius_px = int(min(full_h, full_w) * 0.05)  # Reduced from 0.1 (smaller coins)
    max_radius_px = int(min(full_h, full_w) * 0.95)  # Slightly larger allowance for full coin detection

    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(min_radius_px * 2, 30),  # Reduced from 3 to 2, 50 to 30
        param1=80,   # Reduced from 100 (less strict edge detection)
        param2=30,   # Reduced from 50 (lower accumulator threshold)
        minRadius=min_radius_px,
        maxRadius=max_radius_px,
    )

    # 2. Detect ellipses from contours
    edges = cv2.Canny(gray_blurred, 30, 120)  # Reduced from 50,150 (more lenient edge detection)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    for contour in contours:
        if len(contour) >= 5:  # Need at least 5 points to fit an ellipse
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (axis1, axis2), angle = ellipse

            # Sort axes: major >= minor
            major_axis = max(axis1, axis2)
            minor_axis = min(axis1, axis2)

            # Aspect ratio (how elongated)
            aspect_ratio = major_axis / max(minor_axis, 1e-6)

            # Calculate tilt correction: cos(θ) = minor/major
            tilt_correction = minor_axis / major_axis

            # Use major axis to approximate diameter; filter by size in px
            # major_axis is the projected diameter in pixels
            if min_radius_px * 2 < major_axis < max_radius_px * 2 and aspect_ratio < 3.0:  # Increased from 2.0 to 3.0
                r_approx = int(major_axis / 2)  # radius from major axis
                ellipses.append((int(cx), int(cy), r_approx, ellipse, tilt_correction))

    # 3. Score all candidates (circles + ellipses) using color heuristics
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    best_candidate = None
    best_score = -1.0
    candidates = []

    # Add circles as candidates (tilt_correction = 1.0 for circles)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            # Check if coin is completely within image bounds
            if r <= x < full_w - r and r <= y < full_h - r:
                candidates.append(("circle", x, y, r, None, 1.0))

    # Add ellipses as candidates (with their tilt correction)
    for x, y, r, ellipse_data, tilt_correction in ellipses:
        # Check if coin is completely within image bounds
        if r <= x < full_w - r and r <= y < full_h - r:
            candidates.append(("ellipse", x, y, r, ellipse_data, tilt_correction))

    # Score each candidate
    for candidate_type, x, y, r, extra_data, tilt_correction in candidates:
        # Create a mask for this candidate
        mask = np.zeros(corrected_L.shape, dtype=np.uint8)
        if candidate_type == "circle":
            cv2.circle(mask, (x, y), r, 255, -1)
        else:  # ellipse
            cv2.ellipse(mask, extra_data, 255, -1)

        # Score using the comprehensive scoring function
        score = _score_coin_candidate(
            hsv_img=hsv_img, gray_img=corrected_L, mask=mask, x=x, y=y, r=r, tilt_correction=tilt_correction
        )

        if score > best_score:
            best_score = score
            best_candidate = (candidate_type, x, y, r, tilt_correction)

    # Only accept if score is above threshold (reject weak candidates)
    # Higher threshold since we now check for penny colors and full visibility
    MIN_COIN_SCORE = 0  # Minimum score to consider it a coin
    if best_candidate is not None:
        _, x, y, r, tilt_correction = best_candidate
        coin_radius_cm = 0.9525  # US penny radius in cm
        pixels_per_cm = r / coin_radius_cm

        return pixels_per_cm, (x, y, r), tilt_correction

    return None, None, None


def _score_coin_candidate(
    hsv_img: np.ndarray,
    gray_img: np.ndarray,
    mask: np.ndarray,
    x: int,
    y: int,
    r: int,
    tilt_correction: float,
) -> float:
    """
    Return a scalar score; higher is better.
    hsv_img: original image in HSV
    gray_img: grayscale or corrected_L (for edge detection)
    mask: binary mask of candidate region (uint8, 0/255)
    """

    region = mask > 0
    if not np.any(region):
        return -1.0

    h_vals = hsv_img[:, :, 0][region]
    s_vals = hsv_img[:, :, 1][region]
    v_vals = hsv_img[:, :, 2][region]

    # --- 1. Basic stats ---
    mean_v = float(np.mean(v_vals))
    mean_h = float(np.mean(h_vals))
    std_h = float(np.std(h_vals))
    std_s = float(np.std(s_vals))

    # Brightness in [0,1] - coins are typically brighter than skin lesions
    brightness_score = mean_v / 255.0

    # Uniformity in [0,1] (1 = very uniform)
    uniformity_score = 1.0 - (std_h / 180.0 + std_s / 255.0) / 2.0
    uniformity_score = max(0.0, min(1.0, uniformity_score))
    
    # --- 1b. Coin color detection ---
    # Pennies are typically orange/bronze: hue 5-30 in HSV (expanded range)
    # Also check for copper/bronze colors (hue 0-5 or 25-30 for reddish tones)
    # Reject dark lesions (low brightness) and non-metallic colors
    is_penny_color = False
    
    # Primary: Orange/bronze range (typical penny color)
    if 5 <= mean_h <= 30:
        is_penny_color = True
    # Secondary: Reddish-copper tones (hue 0-5 or 170-180 for red)
    elif (0 <= mean_h <= 5) or (170 <= mean_h <= 180):
        # Check if it's bright enough to be a coin (not a dark lesion)
        if mean_v > 80:  # Coins are brighter than most lesions
            is_penny_color = True
    # Tertiary: Low saturation metallic (silver coins) - but prioritize pennies
    elif std_s < 25 and mean_v > 100:
        is_penny_color = True
    
    # Strong penalty for non-coin colors (lesions are typically darker, different hue)
    if not is_penny_color:
        color_score = 0.1  # Very low score for non-penny colors
    elif mean_v < 60:  # Too dark to be a coin
        color_score = 0.2
    else:
        color_score = 1.0  # Full score for penny-like colors

    # --- 2. Edge strength around boundary ---
    # Build a thin ring at the border of the candidate
    edge_mask = np.zeros_like(mask)
    cv2.circle(edge_mask, (x, y), int(r * 1.05), 255, 2)  # small band
    edge_band = edge_mask > 0

    # Canny edges on gray_img
    edges = cv2.Canny(gray_img, 50, 150)
    edge_vals = edges[edge_band]
    # Normalize edge strength by max 255
    edge_strength = float(np.mean(edge_vals)) / 255.0 if edge_vals.size > 0 else 0.0

    # --- 3. Contrast vs local background ---
    outer_mask = np.zeros_like(mask)
    cv2.circle(outer_mask, (x, y), int(r * 1.5), 255, -1)
    # ring outside the coin: outer minus inner
    ring = (outer_mask > 0) & (~region)

    if np.any(ring):
        h_ring = hsv_img[:, :, 0][ring]
        s_ring = hsv_img[:, :, 1][ring]
        v_ring = hsv_img[:, :, 2][ring]

        dv = abs(mean_v - float(np.mean(v_ring))) / 255.0
        dh = abs(float(np.mean(h_vals)) - float(np.mean(h_ring))) / 180.0
        ds = abs(float(np.mean(s_vals)) - float(np.mean(s_ring))) / 255.0

        contrast_score = (dv + dh + ds) / 3.0
    else:
        contrast_score = 0.0

    # --- 4. Tilt penalty (prefer cosθ close to 1) ---
    # For circles, tilt_correction = 1.0
    # For ellipses, tilt_correction = minor/major = cosθ
    tilt_score = float(tilt_correction)
    tilt_score = max(0.0, min(1.0, tilt_score))

    # --- 5. Size check (penny should be ~1.9cm, reasonable pixel size) ---
    # Typical penny in photo: 50-200 pixels radius depending on distance
    img_h, img_w = hsv_img.shape[:2]
    img_diag = np.sqrt(img_h**2 + img_w**2)

    # --- 6. Combine scores ---
    # Prioritize: color match > edge strength > size > contrast > brightness
    score = (
        0.30 * color_score          # NEW: Strong weight on coin-like colors
        + 0.25 * edge_strength      # Strong circular edge is key
        + 0.15 * contrast_score     # Coin vs skin contrast
        + 0.05 * brightness_score   # Reduced weight
        + 0.03 * uniformity_score   # Reduced (coins can have markings)
        + 0.02 * tilt_score         # Reduced
    )

    return score


def create_coin_masks(
    mask: np.ndarray, downscaled_image: np.ndarray, coin_data: Tuple[int, int, int], roi_x: int, roi_y: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create coin masks and remove coin from segmentation mask.

    Args:
        mask: Segmentation mask to clean (will be modified in place)
        downscaled_image: Full downscaled image
        coin_data: Tuple of (center_x, center_y, radius) in full image coordinates
        roi_x: X offset of crop region
        roi_y: Y offset of crop region

    Returns:
        mask: Updated mask with coin removed
        coin_mask: Coin mask in cropped space
        coin_mask_full: Coin mask in full image space
    """

    coin_mask = np.zeros_like(mask)
    coin_mask_full = np.zeros(downscaled_image.shape[:2], dtype=np.uint8)

    if coin_data is not None:
        full_cx, full_cy, full_cr = coin_data

        # Expand coin radius to fully remove coin edges/reflections
        expanded_radius = int(full_cr * 1.4) + 4

        # Draw on full space mask (downscaled image space)
        cv2.circle(coin_mask_full, (full_cx, full_cy), expanded_radius, 255, -1)

        # Draw on cropped space mask (for segmentation exclusion)
        crop_cx = full_cx - roi_x
        crop_cy = full_cy - roi_y
        cv2.circle(mask, (crop_cx, crop_cy), expanded_radius, 0, -1)  # Remove expanded area
        cv2.circle(coin_mask, (crop_cx, crop_cy), expanded_radius, 255, -1)

        # Dilate masks slightly to ensure full coverage
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        coin_mask_full = cv2.dilate(coin_mask_full, kernel, iterations=1)
        coin_mask = cv2.dilate(coin_mask, kernel, iterations=1)

    return mask, coin_mask, coin_mask_full


# --- END Existing Helper Functions ---


def _calculate_metrics(
    filtered_image: np.ndarray, final_mask: np.ndarray, pixels_per_cm: float = None, tilt_correction: float = 1.0
) -> Dict[str, Any]:
    """Calculates Compactness, Color Stats (LAB), Dominant Hues, and Area.

    Args:
        filtered_image: Preprocessed image
        final_mask: Final segmentation mask
        pixels_per_cm: Pixel-to-cm conversion factor
        tilt_correction: cos(θ) correction factor for tilted coins (default 1.0)
    """

    metrics = {}

    # Convert to LAB color space for color analysis
    roi_image_lab = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2LAB)
    lesion_pixels_lab = roi_image_lab[final_mask > 0]

    if lesion_pixels_lab.size == 0:
        return metrics

    # 1. Compactness (Shape Metric)
    # Compactness Index = Perimeter^2 / (4 * pi * Area)
    # Value close to 1 is highly circular.
    try:
        # Find contour (perimeter)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # We assume the largest component is the only one in the mask
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)

            if area > 0:
                compactness = (perimeter * perimeter) / (4.0 * math.pi * area)
                metrics["compactness_index"] = float(compactness)
            else:
                metrics["compactness_index"] = 0.0
    except Exception:
        metrics["compactness_index"] = np.nan

    # 2. Color Statistics (LAB)
    # L=Lightness, A=Green-Red axis, B=Blue-Yellow axis
    mean_lab = lesion_pixels_lab.mean(axis=0)
    std_lab = lesion_pixels_lab.std(axis=0)

    metrics["color_stats_lab"] = {
        "mean_L": float(mean_lab[0]),
        "mean_A": float(mean_lab[1]),
        "mean_B": float(mean_lab[2]),
        "std_L": float(std_lab[0]),
        "std_A": float(std_lab[1]),
        "std_B": float(std_lab[2]),
    }

    # 3. Dominant Hues (K-Means Clustering on LAB)
    # Find the top 3 dominant colors/hues within the lesion
    K = min(3, len(lesion_pixels_lab))  # Use up to 3 clusters
    if K > 0:
        try:
            kmeans = KMeans(n_clusters=K, n_init="auto", random_state=42).fit(lesion_pixels_lab)
            centroids = kmeans.cluster_centers_
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
        except Exception:
            pass

    # 4. Lesion Area (in cm²)
    if pixels_per_cm:
        pixel_count = cv2.countNonZero(final_mask)
        area_cm2 = pixel_count / (pixels_per_cm**2)

        # Correct for coin tilt if applicable
        # When coin is tilted, the scale might be off. We correct by dividing by cos(θ)
        area_cm2_corrected = area_cm2 / tilt_correction if tilt_correction > 0 else area_cm2

        metrics["area_cm2"] = area_cm2_corrected
        metrics["area_cm2_uncorrected"] = area_cm2
        metrics["tilt_correction_factor"] = tilt_correction
    else:
        metrics["area_cm2"] = None
        metrics["area_cm2_uncorrected"] = None
        metrics["tilt_correction_factor"] = None

    return metrics


def crop_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int], initial_scale: float) -> np.ndarray:
    """Crop the image to the bounding box."""
    x, y, w, h = bbox
    roi_x, roi_y = 0, 0
    if bbox:
        # Scale bbox to match the downscaled image
        x, y, w, h = bbox
        x = int(x * initial_scale)
        y = int(y * initial_scale)
        w = int(w * initial_scale)
        h = int(h * initial_scale)

        h_img, w_img = image.shape[:2]
        x = max(0, min(x, w_img))
        y = max(0, min(y, h_img))
        w = max(0, min(w, w_img - x))
        h = max(0, min(h, h_img - y))
        if w > 0 and h > 0:
            image = image[y : y + h, x : x + w]
            roi_x, roi_y = x, y
    return image, roi_x, roi_y, (x, y, w, h)


def create_bad_color_mask(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify and filter non-skin colors and inpaint highlights.

    Args:
        image: Input BGR image

    Returns:
        filtered_image: Image with non-skin colors removed and highlights inpainted
        bad_color_mask: Boolean mask of non-skin colors
    """
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    # Mark as bad: non-skin colors (greens, blues, purples)
    # Keep dark regions - they're part of melanoma
    # Made less strict: increased saturation threshold and narrowed hue range
    bad_color_mask = (s > 80) & (v > 100) & ((h > 30) & (h < 140))  # More lenient

    filtered_image = image.copy()
    filtered_image[bad_color_mask] = 0

    # Inpaint specular highlights
    highlights = (s < 40) & (v > 200)
    if np.any(highlights):
        highlight_mask = highlights.astype(np.uint8) * 255
        highlight_mask = cv2.dilate(highlight_mask, None, iterations=2)
        filtered_image = cv2.inpaint(filtered_image, highlight_mask, 3, cv2.INPAINT_TELEA)

    return filtered_image, bad_color_mask


def correct_illumination(image: np.ndarray) -> np.ndarray:
    """
    Correct uneven illumination in the image.

    Args:
        image: Input BGR image

    Returns:
        corrected_L: Illumination-corrected luminance channel (grayscale)
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Estimate background illumination using morphological closing
    kernel_size = _adaptive_block_size(L.shape) * 4
    if kernel_size % 2 == 0:
        kernel_size += 1
    bg_illumination = cv2.morphologyEx(
        L, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    )

    # Normalize by background illumination
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected_L = (L.astype(np.float32) / bg_illumination.astype(np.float32)) * 255.0
    corrected_L = np.clip(corrected_L, 0, 255).astype(np.uint8)

    # Smooth and enhance contrast
    corrected_L = cv2.GaussianBlur(corrected_L, (5, 5), 0)
    corrected_L = _maybe_enhance_contrast(corrected_L)

    return corrected_L


def threshold_to_mask(corrected_L: np.ndarray, bad_color_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create initial segmentation mask using thresholding and morphological operations.

    Args:
        corrected_L: Illumination-corrected luminance channel
        bad_color_mask: Mask of regions to exclude

    Returns:
        mask: Initial segmentation mask
        kernel: Morphological kernel used
    """
    # Threshold luminance using Otsu's method
    _, mask = cv2.threshold(corrected_L, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask[bad_color_mask] = 0

    # Fallback to adaptive threshold if Otsu fails
    filled_ratio = cv2.countNonZero(mask) / mask.size
    if filled_ratio < 0.01 or filled_ratio > 0.99:
        block_size = _adaptive_block_size(corrected_L.shape)
        mask = cv2.adaptiveThreshold(
            corrected_L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 2
        )
        mask[bad_color_mask] = 0

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask, kernel


def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline: filter, correct illumination, and create initial mask.

    Args:
        image: Input BGR image

    Returns:
        filtered_image: Image with non-skin colors removed and highlights inpainted
        bad_color_mask: Boolean mask of non-skin colors
        mask: Initial segmentation mask after thresholding and morphological operations
        kernel: Morphological kernel for further processing
    """
    # A) Create bad color mask and filter image
    filtered_image, bad_color_mask = create_bad_color_mask(image)

    # B) Correct illumination
    corrected_L = correct_illumination(filtered_image)

    # C) Threshold to create initial mask
    mask, kernel = threshold_to_mask(corrected_L, bad_color_mask)

    return filtered_image, bad_color_mask, mask, kernel


def refine_mask_with_grabcut(
    filtered_image: np.ndarray, mask: np.ndarray, bad_color_mask: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    """
    Refine segmentation mask using GrabCut algorithm.

    Args:
        filtered_image: Preprocessed image
        mask: Initial segmentation mask
        bad_color_mask: Mask of non-skin colors
        kernel: Morphological kernel

    Returns:
        refined_mask: Refined segmentation mask
    """
    # 1. Define Sure FG from our conservative threshold mask
    sure_fg = cv2.erode(mask, kernel, iterations=3)

    # 2. Define Probable FG by dilating the initial mask slightly
    prob_fg = cv2.dilate(mask, kernel, iterations=5)

    # 3. Initialize mask for GrabCut
    # 0=BG, 1=FG, 2=Prob BG, 3=Prob FG
    gc_mask = np.zeros(mask.shape, np.uint8)
    gc_mask[:] = cv2.GC_BGD  # Default everything to Sure BG

    # Set probable foreground from dilated mask
    gc_mask[prob_fg > 0] = cv2.GC_PR_FGD

    # Pin our thresholded core as Sure FG (helps it anchor)
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    # Pin bad colors (markers) as Sure BG
    gc_mask[bad_color_mask] = cv2.GC_BGD

    # 3. Run GrabCut using MASK mode
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(filtered_image, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        pass

    refined_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    return refined_mask


def run_cv_analysis(image_path: str, bbox: Tuple[int, int, int, int] = None) -> Dict[str, Any]:
    """
    Segment a skin lesion and compute diagnostic metrics.

    Args:
        image_path: Path to the image file.
        bbox: Optional (x, y, w, h) bounding box. If None, uses full image.
    """
    # Load image and immediately downscale to max 1024x1024 if needed
    full_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    #Save image to downloads
    cv2.imwrite('/Users/tk20/Downloads/tmp.png', full_image)

    # Initialize bbox with full image boundaries if not provided
    full_h, full_w = full_image.shape[:2]
    if bbox is None:
        bbox = (0, 0, full_w, full_h)

    # Downscale the image to max 1024x1024 if needed, to save processing time
    image, initial_scale = _downscale_if_needed(full_image)

    # Crop the image to the bounding box for segmentation
    cropped_image, roi_x, roi_y, bbox_scaled = crop_bbox(image, bbox, initial_scale)
    cropped_height, cropped_width = cropped_image.shape[:2]

    # Preprocess: filter, correct illumination, and create initial mask
    filtered_image, bad_color_mask, initial_mask, kernel = preprocess_image(cropped_image)

    # GrabCut refinement
    grabcut_mask = refine_mask_with_grabcut(filtered_image, initial_mask, bad_color_mask, kernel)

    # Remove coin from mask and create coin masks
    pixels_per_cm, coin_data, tilt_correction = _detect_coin_and_get_scale(image)
    final_mask, coin_mask, coin_mask_full = create_coin_masks(grabcut_mask, image, coin_data, roi_x, roi_y)

    # Clean up small artifacts (e.g., hairs) and keep the largest component
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_clean)
    final_mask = _largest_component(final_mask)

    # --- METRICS CALCULATION ---
    metrics = _calculate_metrics(filtered_image, final_mask, pixels_per_cm, tilt_correction if tilt_correction else 1.0)

    return {
        "metrics": metrics,  # Includes area_cm2 (tilt-corrected if applicable)
        "bbox": bbox_scaled,
        "coin_data": coin_data,
        "pixels_per_cm": pixels_per_cm,
        "tilt_correction": tilt_correction,
        "images": {
            "original_image": image,
            "cropped_image": cropped_image,
            "filtered_image": filtered_image,
        },
        "masks": {
            "grabcut_mask": grabcut_mask.astype(np.uint8),
            "coin_mask": coin_mask.astype(np.uint8),
            "coin_mask_full": coin_mask_full.astype(np.uint8),
            "final_mask": final_mask.astype(np.uint8),
            "initial_mask": initial_mask.astype(np.uint8),
            "bad_color_mask": bad_color_mask.astype(np.uint8),
        },
    }
