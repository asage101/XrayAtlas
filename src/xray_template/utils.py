import matplotlib.pyplot as plt
import numpy as np

def display_image(image, title="Image"):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()


def compute_difference_map(template, registered_xray):
    diff_map = np.abs(template - registered_xray)
    plt.imshow(diff_map, cmap="jet")
    plt.colorbar()
    plt.title("Difference Map")
    plt.axis("off")
    plt.show()
