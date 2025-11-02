import matplotlib.pyplot as plt
from PIL import Image
from devernay_edges import DevernayEdges

if __name__ == "__main__":
    sigma = 3.0
    high_treshold = 4.0
    low_threshold = 0.0

    image_rgb = Image.open("cat.jpg")
    image_rgb_gray = image_rgb.convert('L')
    image_binary = image_rgb_gray.point(lambda x: 0 if x < 240 else 255, '1')

    [edges_x, edges_y] = DevernayEdges(image_binary, sigma, high_treshold, low_threshold).run()

    plt.figure(1)
    plt.title("Devernay Edge Detection")
    plt.imshow(image_rgb)
    plt.scatter(edges_x, edges_y, color="magenta", marker=".", linewidth=.1)
    plt.show()
