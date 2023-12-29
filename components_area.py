import numpy as np
import cv2

from utils import plot_image


def detect_defects_within_components_area(img1, img2, kernel_size=(3,3),kernel_type=cv2.MORPH_RECT):
    # Apply Gaussian Blur to both images
    blur1 = cv2.GaussianBlur(img1, (5, 5), 0)
    blur2 = cv2.GaussianBlur(img2, (5,5), 0)

    # Apply Otsu's thresholding
    _, thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Calculate the absolute difference
    diff = cv2.absdiff(thresh1, thresh2)
    _, diff = cv2.threshold(diff, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # Apply morphological closing to the difference
    kernel = cv2.getStructuringElement(kernel_type, kernel_size)
    closing = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)

    _, res = cv2.threshold(closing, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    return res

def extract_component_area(ref_image,kernel_type=cv2.MORPH_RECT,kernel_size=(10,10),n_times_eroision=11):
    reference_image=ref_image
    for i in range(10):
        reference_image = cv2.medianBlur(reference_image, 5)
    _, thresh_reference_image = cv2.threshold(reference_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(kernel_type, kernel_size)
    img=thresh_reference_image
    for i in range(n_times_eroision):
        # Apply erosion
        #img = cv2.erode(img, kernel)
        # Apply closing
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


def is_border_pixel(img, x, y):
    """ Check if the pixel at (x, y) is a border pixel in the binary image """
    # Define the neighbors (8-connectivity)
    neighbors = [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
                 (x - 1, y), (x + 1, y),
                 (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]

    whites = blacks = 0
    for nx, ny in neighbors:
        if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:  # Check if within bounds
            if img[nx, ny] == 255:
                whites += 1
            else:
                blacks += 1

    return whites > 0 and blacks > 0


def create_new_image(img1, img2):
    """ Create a new image based on the specified conditions """
    new_image = np.copy(img1)

    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            if img1[x, y] == 255 and is_border_pixel(img2, x, y):
                new_image[x, y] = 0

    return new_image
