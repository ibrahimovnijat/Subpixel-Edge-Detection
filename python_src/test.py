
import matplotlib.pyplot as plt 
from PIL import Image

from devernay_edges import DevernayEdges


def main():
    sigma = 3.0
    high_treshold = 4.0
    low_threshold = 0.0

    image_rgb = Image.open("cat.jpg")    
    # image_rgb = Image.open("dog.jpg")

    # convert to binary
    image_binary = Image.open("binary_cat_image.jpg")
    # image_binary = Image.open("binary_dog_image.jpg")

    devernayEdges = DevernayEdges(image_binary, sigma, high_treshold, low_threshold)
    [edges_x, edges_y] = devernayEdges.detect_edges()
    
    print(f"edges_x len: {len(edges_x)}")
    print(f"edges_y len: {len(edges_y)}")

    plt.figure(1)
    plt.title("Devernay Edge Detection")
    plt.imshow(image_rgb)
    plt.scatter(edges_x, edges_y, color="magenta", marker=".", linewidth=.1)
    plt.show()


if __name__ == "__main__": main()