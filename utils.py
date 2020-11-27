import difflib

import numpy as np
from PIL import Image
import scipy.ndimage
import scipy.signal
import scipy.stats


def threshold_image(image_np, threshold=0, op = '<'):
    # Set pixels with value less than threshold to 0, otherwise set is as 255
    if op == '<':
        image_result_np = np.where(image_np < threshold, 0, 1)
    else:
        image_result_np = np.where(image_np > threshold, 0, 1)
    # Convert numpy array back to PIL image object
    image_result = Image.fromarray((image_result_np * 255).astype(np.uint8))
    return image_result


def otsu_thresholding_in(image, max_value=255):
    # Image must be in grayscale
    image_np = np.array(image)
    # Set total number of bins in the histogram
    number_of_bins = 256  # Since our image is 8 bits, we used 256 for now
    # Get the image histogram
    histogram, bin_edges = np.histogram(image_np, bins=number_of_bins)

    # Calculate centers of bins
    bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2.
    # Iterate over all thresholds (indices) and get the probabilities \w_0(t), \w_1(t)
    w_0 = np.cumsum(histogram)
    w_1 = np.cumsum(histogram[::-1])[::-1]

    # Get the class means \mu0(t)
    m_0 = np.cumsum(histogram * bin_center) / w_0
    # Get the class means \mu1(t)
    m_1 = (np.cumsum((histogram * bin_center)[::-1]) / w_1[::-1])[::-1]

    # Calculate the inter-class variance
    inter_var = w_0[:-1] * w_1[1:] * (m_0[:-1] - m_1[1:]) ** 2

    # Minimize intra-class variance, which is equal to maximize the inter_class_variance function val
    max_val_index = np.argmax(inter_var)

    # Get the threshold value
    thresh = bin_center[:-1][max_val_index]
    # Get the image by performing the thresholding
    image_result = threshold_image(image_np, thresh)

    return image_result, thresh


# https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
def gaussian_kernel(kernel_size=7, std=1, normalize=True):
    gaussian_kernel_1d = scipy.signal.gaussian(kernel_size, std=std).reshape(kernel_size, 1)
    gaussian_kernel_2d = np.outer(gaussian_kernel_1d, gaussian_kernel_1d)
    if normalize:
        return gaussian_kernel_2d / gaussian_kernel_2d.sum()
    else:
        return gaussian_kernel_2d

# https://www.mathworks.com/matlabcentral/fileexchange/8647-local-adaptive-thresholding
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/adpthrsh.htm
def adaptive_gaussian_thresholding_in(image, max_value=255, block_size=7, C=0, std=1):
    # Image must be in grayscale
    image_np = np.array(image)

    kernel = gaussian_kernel(block_size, std=std)
    # print(f"kernel={kernel}")

    image_convolved_np = scipy.signal.convolve2d(image_np, kernel, mode='same', boundary='symm')
    image_result_np = image_convolved_np - image_np - C
    # print(image_result_np)

    image_result = threshold_image(image_result_np)

    return image_result


# https://www.mathworks.com/matlabcentral/fileexchange/8647-local-adaptive-thresholding
def adaptive_mean_thresholding_in(image, max_value=255, block_size=7, C=0):
    # Image must be in grayscale
    image_np = np.array(image)

    kernel = np.ones((block_size, block_size)) / (block_size ** 2)
    image_convolved_np = scipy.signal.convolve2d(image_np, kernel, mode='same', boundary='symm')
    image_result_np = image_convolved_np - image_np - C
    image_result = threshold_image(image_result_np)

    return image_result


def evaluate(actual, expected, print_score=True):
    s = difflib.SequenceMatcher(None, actual, expected)
    if print_score:
        print("{:.5f}".format(s.ratio()))
    # print(s.get_matching_blocks())
    return s.ratio()
