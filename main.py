import numpy as np
import argparse
import cv2

from algorithms.alignment import  align_images
from algorithms.components_area import detect_defects_within_components_area
from algorithms.outComponent_area import detect_defects_outside_component_area
from utils import save_image, load_image
import cv2
import matplotlib.pyplot as plt
from utils import plot_image



import cv2
import matplotlib.pyplot as plt

def process_images(reference_path, inspected_path, output_path='results.png', plot_intermidiate_results=False, kernel_size=(3,3), kernel_type=cv2.MORPH_RECT):
    """
    Process images to detect defects and save the results.

    Args:
        reference_path (str): Path to the reference image.
        inspected_path (str): Path to the inspected image.
        output_path (str, optional): Path to save the results image. Defaults to 'results.png'.
        plot_intermidiate_results (bool, optional): Whether to plot intermediate results. Defaults to False.
        kernel_size (tuple, optional): Size of the kernel for morphological operations. Defaults to (3,3).
        kernel_type (int, optional): Type of the kernel for morphological operations. Defaults to cv2.MORPH_RECT.

    Returns:
        numpy.ndarray: Result image with detected defects.
    """
    # Load images
    reference_image = cv2.imread(reference_path)
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    inspected_image = cv2.imread(inspected_path)
    inspected_image = cv2.cvtColor(inspected_image, cv2.COLOR_BGR2GRAY)
    
    # Align images

    aligned_inspected_image = align_images(inspected_image, reference_image)
    reference_image[aligned_inspected_image == 0] = 0
    aligned_reference_image = reference_image
    
    # Detect defects within component area
    defects_in_component_area = detect_defects_within_components_area(aligned_reference_image, aligned_inspected_image)
    
    # Detect defects outside component area
    defects_out_component_area = detect_defects_outside_component_area(aligned_reference_image, aligned_inspected_image)
    
    # Apply morphological dilation
    kernel = cv2.getStructuringElement(kernel_type, kernel_size)
    defects_out_component_area = cv2.morphologyEx(defects_out_component_area, cv2.MORPH_DILATE, kernel)
    
    if plot_intermidiate_results:
        # Plot defects_in_component_area and defects_out_component_area images side by side
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].imshow(defects_in_component_area, cmap='gray')
        axs[0].set_title('defects_in_component_area')
        axs[1].imshow(defects_out_component_area, cmap='gray')
        axs[1].set_title('defects_out_component_area')
        plt.show()
    
    # Combine defect images
    results = cv2.bitwise_or(defects_in_component_area, defects_out_component_area * 255)
    
    # Save results
    save_image(results, output_path)
    
    # Show results
    plt.imshow(results, cmap='gray')
    plt.show()
    
    return results



def main():
    # Hardcoded paths for debugging
    inspected_path = "data/defective_examples/case2_inspected_image.tif"
    reference_path = r"data/defective_examples/case2_reference_image.tif"
    process_images(reference_path, inspected_path,plot_intermidiate_results=True)

if __name__ == "__main__":
    main()