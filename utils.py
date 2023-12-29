import cv2
import matplotlib.pyplot as plt

def load_image(image_path):
    """
    Load an image from the specified path.
    """
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def save_image(img, image_path):

    cv2.imwrite(image_path, img)


    return None


#ploting with title
def plot_image(image, title=""):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()
