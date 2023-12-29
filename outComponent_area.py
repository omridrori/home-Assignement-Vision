from typing import Union
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest

from algorithms.components_area import extract_component_area
from utils import plot_image

import cv2
import numpy as np
import copy


def detect_defects_outside_component_area(ref_image, inspected_image, algorithm='mean', threshold=3):
    """
    Detects defects outside the component area in the inspected image using the specified algorithm.

    Args:
        ref_image (numpy.ndarray): The reference image.
        inspected_image (numpy.ndarray): The inspected image.
        algorithm (str, optional): The algorithm to use for defect detection. Defaults to 'mean'.
        threshold (int, optional): The threshold value for the 'mean' algorithm. Defaults to 3.

    Returns:
        numpy.ndarray: The image with defects outside the component area.
    """

    out_component_area = process_images(ref_image, inspected_image)
    out_component_area_ = out_component_area

    algorithm_mapping = {
        'isolation_forest1': detect_anomalies_with_isolation_forest,
        'mean': detect_anomalies_mean,
        'one_class_svm': detect_anomalies_with_one_class_svm
    }

    if algorithm not in algorithm_mapping:
        raise ValueError(f"Invalid algorithm: {algorithm}. Please choose from {list(algorithm_mapping.keys())}.")

    out_component_area_ = algorithm_mapping[algorithm](out_component_area_, threshold=threshold)

    return out_component_area_.astype('uint8')





def extract_out_component_area(ref_image: np.ndarray, inspected_image: np.ndarray) -> np.ndarray:
    """
    Extracts the pixels in `inspected_image` that are not in `ref_image`.

    Args:
        ref_image (numpy.ndarray): The reference image.
        inspected_image (numpy.ndarray): The image to extract pixels from.

    Returns:
        numpy.ndarray: The extracted pixels.
    """
    component_area = extract_component_area(ref_image)
    inverted_mask = cv2.bitwise_not(component_area)
    extracted_pixels = cv2.bitwise_and(inspected_image, inverted_mask)
    return extracted_pixels

def detect_anomalies_mean(image, threshold=3):
    """
    Detect anomalies in an image based on mean and standard deviation.

    Args:
        image (numpy.ndarray): The input image.
        threshold (float): The threshold for considering a pixel as an anomaly. Default is 3.

    Returns:
        numpy.ndarray: A mask of anomalies.

    """
    # Calculate the mean and standard deviation of the image using numpy functions
    mean = np.mean(image)
    std_dev = np.std(image)

    # Define a threshold beyond which a pixel will be considered an anomaly
    anomaly_threshold = mean + (threshold * std_dev)

    # Create a mask of anomalies
    anomalies = image > anomaly_threshold

    return anomalies



def detect_anomalies_with_isolation_forest(image: np.ndarray, n_estimators: int = 100, contamination: float = 0.01) -> np.ndarray:
    """
    Detects anomalies in the given image using Isolation Forest algorithm.

    Args:
        image (numpy.ndarray): The input image.
        n_estimators (int, optional): The number of base estimators in the ensemble. Defaults to 100.
        contamination (float, optional): The proportion of outliers in the data set. Defaults to 0.01.

    Returns:
        numpy.ndarray: The mask indicating the anomalies in the image.
    """
    if not isinstance(n_estimators, int) or n_estimators <= 0:
        raise ValueError("Invalid value for n_estimators. Please provide a positive integer.")

    if not isinstance(contamination, float) or contamination < 0 or contamination > 1:
        raise ValueError("Invalid value for contamination. Please provide a float between 0 and 1.")

    # Create a mask for non-zero pixels
    non_zero_mask = image > 0

    # Flatten the image and filter only non-zero pixels
    X = image[non_zero_mask].reshape(-1, 1)

    # Initialize Isolation Forest
    iso_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination)

    # Fit the model on non-zero pixels
    iso_forest.fit(X)

    # Predict anomalies only on non-zero pixels
    anomalies = iso_forest.predict(X)

    # Prepare an empty mask to fill in the anomalies
    anomaly_mask = np.zeros_like(image, dtype=np.uint8)

    # Mark the anomalies in the mask
    anomaly_mask[non_zero_mask] = (anomalies == -1).astype(np.uint8) * 255

    return anomaly_mask


from sklearn.svm import OneClassSVM

# Additional function for anomaly detection using One-Class SVM
def detect_anomalies_with_one_class_svm(image, nu=0.03, gamma='auto', kernel_size=(3, 3), iterations=1):
    """
    Detect anomalies in an image using One-Class SVM.

    Args:
        image (numpy.ndarray): Input image.
        nu (float, optional): An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Defaults to 0.03.
        gamma (float or str, optional): Kernel coefficient for 'rbf'. If 'auto', it uses 1 / n_features. Defaults to 'auto'.
        kernel_size (tuple, optional): Size of the kernel for erosion operation. Defaults to (3, 3).
        iterations (int, optional): Number of times erosion operation is applied. Defaults to 1.

    Returns:
        numpy.ndarray: Anomaly mask indicating the anomalies in the image.
    """
    # Flatten the image and get the pixel intensity values as features
    X = image.reshape(-1, 1)

    # Validate nu and gamma parameters
    if nu <= 0 or nu >= 1:
        raise ValueError("Invalid value for nu parameter. Please choose a value between 0 and 1.")
    if gamma != 'auto' and (gamma <= 0 or gamma >= 1):
        raise ValueError("Invalid value for gamma parameter. Please choose a value between 0 and 1 or 'auto'.")

    # Initialize One-Class SVM
    one_class_svm = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)

    try:
        # Fit the model
        one_class_svm.fit(X)

        # Predict anomalies
        anomalies = one_class_svm.predict(X)
        anomaly_mask = anomalies == -1

        # Reshape the anomaly mask to the original image shape
        anomaly_mask = anomaly_mask.reshape(image.shape)
        anomaly_mask = anomaly_mask.astype(np.uint8)
        kernel = np.ones(kernel_size, np.uint8)
        anomaly_mask = cv2.erode(anomaly_mask, kernel, iterations=iterations)
        return anomaly_mask
    except Exception as e:
        print(f"Error occurred during fitting and prediction: {e}")
        return None



def denoise_and_binarize(image, enable_denoising=False, h=7, templateWindowSize=7, searchWindowSize=21, threshold_type=cv2.THRESH_OTSU | cv2.THRESH_BINARY):
    """
    Denoises and binarizes the given image.

    Args:
        image (numpy.ndarray): The input image.
        enable_denoising (bool): Whether to enable denoising. Defaults to False.
        h (int): Parameter for denoising. Defaults to 7.
        templateWindowSize (int): Parameter for denoising. Defaults to 7.
        searchWindowSize (int): Parameter for denoising. Defaults to 21.
        threshold_type (int): The threshold type for binarization. Defaults to cv2.THRESH_OTSU | cv2.THRESH_BINARY.

    Returns:
        numpy.ndarray: The binarized image.
    """
    if enable_denoising:
        denoised_img = cv2.fastNlMeansDenoising(image, None, h=h, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
    else:
        denoised_img = image

    _, binary_img = cv2.threshold(denoised_img, 0, 255, threshold_type)

    return binary_img

def apply_morphological_smoothing(binary_img, kernel_dim=(2, 2), perform_smoothing=True):
    """
    Apply morphological smoothing to a binary image.

    Args:
        binary_img (numpy.ndarray): The binary image to be smoothed.
        kernel_dim (tuple, optional): The dimensions of the kernel used for smoothing. Defaults to (2, 2).
        perform_smoothing (bool, optional): Whether to perform the smoothing. Defaults to True.

    Returns:
        numpy.ndarray: The smoothed image if perform_smoothing is True, otherwise the original binary image.
    """
    if perform_smoothing:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_dim)
        smoothed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        return smoothed_img
    else:
        return binary_img

def exclude_and_discard_pixels(input_img, binary_img):
    input_img = cv2.GaussianBlur(input_img, (3, 3), 0)
    _, input_img = cv2.threshold(input_img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    rows, cols = input_img.shape[:2]

    inverted_img = cv2.threshold(copy.deepcopy(input_img), 0, 255, cv2.THRESH_BINARY_INV)[1]
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilated_img = cv2.morphologyEx(inverted_img, cv2.MORPH_DILATE, dilation_kernel)

    for i in range(rows):
        for j in range(cols):
            if dilated_img[i, j] == 255:
                binary_img[i, j] = 0  # Directly discard the pixel

    return binary_img

def process_images(image_ref, image_target):
    difference = cv2.absdiff(image_target, image_ref)

    processed_diff = denoise_and_binarize(difference, enable_denoising=True)
    processed_diff = exclude_and_discard_pixels(image_ref, processed_diff)

    return apply_morphological_smoothing(processed_diff, perform_smoothing=True)











