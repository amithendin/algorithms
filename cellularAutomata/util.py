import numpy as np
import matplotlib.pyplot as plt

def display_heatmap(matrix):
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap of 2D Matrix')
    plt.show()

def convolve2d(image, kernel, padding='same'):
    # Get dimensions of the image and kernel
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the output dimensions with padding
    if padding == 'same':
        output_height = img_height
        output_width = img_width
        pad_vertical = (kernel_height - 1) // 2
        pad_horizontal = (kernel_width - 1) // 2
    else:
        output_height = img_height - kernel_height + 1
        output_width = img_width - kernel_width + 1
        pad_vertical = 0
        pad_horizontal = 0

    # Apply zero-padding to the input image
    padded_image = np.pad(image, ((pad_vertical, pad_vertical), (pad_horizontal, pad_horizontal)), mode='constant')

    # Initialize the output matrix
    output = np.zeros((output_height, output_width))

    # Perform 2D convolution
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output