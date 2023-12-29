import cv2

def apply_threshold(image, threshold_value=127, max_value=255):
    """
    Applies a simple thresholding operation to the input image.

    Args:
        image (numpy.ndarray): The input grayscale image to process.
        threshold_value (int): Threshold value used for binarizing the image.
        max_value (int): Maximum value to use with the THRESH_BINARY thresholding type.

    Returns:
        numpy.ndarray: The binary image after applying thresholding.
    """
    # Convert image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the threshold
    _, thresholded_image = cv2.threshold(image, threshold_value, max_value, cv2.THRESH_BINARY)

    return thresholded_image
