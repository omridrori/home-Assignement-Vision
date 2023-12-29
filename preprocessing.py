import skimage.exposure
import numpy as np
import skimage


def align_image_histograms(source_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
    """
    Aligns the histogram of the source image to match the histogram of the target image.

    Parameters:
        source_img (np.ndarray): The source image.
        target_img (np.ndarray): The target image.

    Returns:
        np.ndarray: The aligned image with matched histograms.
    """
    # Align the histogram of the source image to match the histogram of the target image
    aligned_colors_image = skimage.exposure.match_histograms(source_img, target_img).astype(np.uint8)
    return aligned_colors_image


