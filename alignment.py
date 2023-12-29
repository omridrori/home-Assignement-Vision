import numpy as np
import cv2
from typing import Sequence, Set, Tuple

def extract_and_match_features(source_image: np.ndarray, destination_image: np.ndarray, feature_count: int) -> Tuple[Sequence[cv2.KeyPoint], Sequence[cv2.KeyPoint], list[cv2.DMatch]]:
    """
    Extracts and matches features between two images using ORB algorithm.

    Args:
        source_image: The source image.
        destination_image: The destination image.
        feature_count: The number of desired features.

    Returns:
        A tuple containing the keypoints of the source image, keypoints of the destination image, and the sorted matches.
    """
    orb_detector = cv2.ORB_create(feature_count)  # Create an ORB detector with the specified feature count
    source_keypoints, source_descriptors = orb_detector.detectAndCompute(source_image, None)  # Detect and compute keypoints and descriptors for the source image
    destination_keypoints, destination_descriptors = orb_detector.detectAndCompute(destination_image, None)  # Detect and compute keypoints and descriptors for the destination image
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Create a Brute-Force Matcher with Hamming distance
    matches = matcher.match(source_descriptors, destination_descriptors)  # Match the descriptors of the source and destination images
    sorted_matches = sorted(matches, key=lambda x: x.distance)  # Sort the matches based on their distance
    return source_keypoints, destination_keypoints, sorted_matches
import cv2
import numpy as np

def estimate_transform_matrix(source_keypoints, destination_keypoints, matches):
    """
    Estimates a transformation matrix between two sets of keypoints.

    Args:
    - source_keypoints: A sequence of cv2.KeyPoint objects representing the keypoints in the source image.
    - destination_keypoints: A sequence of cv2.KeyPoint objects representing the keypoints in the destination image.
    - matches: A list of cv2.DMatch objects representing the matches between source and destination keypoints.

    Returns:
    - transform_matrix: A numpy array representing the estimated transformation matrix.
    """

    source_pts = np.float32([source_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    destination_pts = np.float32([destination_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    transform_matrix, mask = cv2.estimateAffinePartial2D(source_pts, destination_pts)
    return transform_matrix


def apply_transform_to_image(image: np.ndarray, transform_matrix: np.ndarray, dimensions: Tuple[int, int]) -> np.ndarray:
    transformed_image = cv2.warpAffine(image, transform_matrix, dimensions, flags=cv2.INTER_CUBIC)
    return transformed_image

def create_alignment_mask(transform_matrix: np.ndarray, dimensions: Tuple[int, int]) -> np.ndarray:
    base_image = np.ones((dimensions[1], dimensions[0]), dtype=np.float32)  # rows, cols
    alignment_mask = cv2.warpAffine(base_image, transform_matrix, (dimensions[0], dimensions[1]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
    return alignment_mask

def identify_non_corresponding_regions(mask_image: np.ndarray) -> Set[Tuple[int, int]]:
    """
    Identifies non-corresponding regions in an alignment mask.

    Args:
        mask_image: The alignment mask.

    Returns:
        A set of non-corresponding regions.
    """
    non_corresponding_region = set()
    for row in range(mask_image.shape[0]):
        for col in range(mask_image.shape[1]):
            if mask_image[row, col] == 0:
                non_corresponding_region.add((row, col))
    return non_corresponding_region

def align_images(source_image: np.ndarray, destination_image: np.ndarray, feature_count: int = 1000) -> np.ndarray:
    source_keypoints, destination_keypoints, sorted_matches = extract_and_match_features(source_image, destination_image, feature_count)

    transform_matrix = estimate_transform_matrix(source_keypoints, destination_keypoints, sorted_matches)

    # Apply transformation
    dimensions = (destination_image.shape[1], destination_image.shape[0])  # cols, rows
    aligned_image = apply_transform_to_image(source_image, transform_matrix, dimensions)

    # Identify non-corresponding regions
    alignment_mask = create_alignment_mask(transform_matrix, dimensions)
    non_corresponding_regions = identify_non_corresponding_regions(alignment_mask)

    # Additional processing can be done here if needed, such as handling non-corresponding regions

    return aligned_image