#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2 as cv
import numpy
from PIL import Image
import scipy.ndimage
import scipy.signal
import pytesseract
import difflib
import os
import numpy as np


# In[3]:


image_folder = "./images"
text_folder = "./source"
images = ["sample01.png", "sample02.png"]
texts = ["sample01.txt", "sample02.txt"]


# In[4]:


def evaluate(actual, expected, print_score=True):
    s = difflib.SequenceMatcher(None, actual, expected)
    if print_score:
        print("{:.5f}".format(s.ratio()))
    # print(s.get_matching_blocks())
    return s.ratio()


# # Base Image with OCR

# In[5]:


for idx, image_name in enumerate(images):
    image = Image.open(os.path.join(image_folder, image_name))
    print(image.format, image.mode)
    image = image.convert("RGB")
    result = pytesseract.image_to_string(image)

    with open(os.path.join(text_folder, texts[idx]), 'r') as f:
            base_text = f.readlines()
            base_text = "".join(base_text)
            # base_text = [line.strip() for line in base_text]


    print(result)
    evaluate(result, base_text)


# # Otsu thresholding

# In[5]:


def threshold_image(image_np, threshold=0, op = '<'):
    # Set pixels with value less than threshold to 0, otherwise set is as 255
    if op == '<':
        image_result_np = np.where(image_np < threshold, 0, 1)
    else:
        image_result_np = np.where(image_np > threshold, 0, 1)
    # Convert numpy array back to PIL image object
    image_result = Image.fromarray((image_result_np * 255).astype(np.uint8))
    return image_result


# In[6]:


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


# In[89]:


for idx, image_name in enumerate(images):
    image = Image.open(os.path.join(image_folder, image_name))
    # print(image.format, image.mode)
    image = image.convert("L")

    with open(os.path.join(text_folder, texts[idx]), 'r') as f:
            base_text = f.readlines()
            base_text = "".join(base_text)
            # base_text = [line.strip() for line in base_text]
    img_cv = numpy.array(image)
    ret, image_th_cv = cv.threshold(img_cv, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    image_th = Image.fromarray(image_th_cv)
    result_th = pytesseract.image_to_string(image_th)
    image_th.show()

    evaluate(result_th, base_text)


# ### Self implementation of Otsu thresholding

# In[58]:


for idx, image_name in enumerate(images):
    image = Image.open(os.path.join(image_folder, image_name))
    # print(image.format, image.mode)
    image = image.convert("L")

    with open(os.path.join(text_folder, texts[idx]), 'r') as f:
            base_text = f.readlines()
            base_text = "".join(base_text)
            # base_text = [line.strip() for line in base_text]

    image_th, thresh = otsu_thresholding_in(image)
    print(f"Threshold pixel value={thresh}")
    image_th.show()
    result_th = pytesseract.image_to_string(image_th)
    

    evaluate(result_th, base_text)


# # Adaptive Gaussian

# In[7]:


# https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
def gaussian_kernel(kernel_size=7, std=1, normalize=True):
    gaussian_kernel_1d = scipy.signal.gaussian(kernel_size, std=std).reshape(kernel_size, 1)
    gaussian_kernel_2d = np.outer(gaussian_kernel_1d, gaussian_kernel_1d)
    if normalize:
        return gaussian_kernel_2d / gaussian_kernel_2d.sum()
    else:
        return gaussian_kernel_2d


# In[8]:


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

    image_result = threshold_image(image_result_np, op='>')

    return image_result


# https://www.mathworks.com/matlabcentral/fileexchange/8647-local-adaptive-thresholding
def adaptive_mean_thresholding_in(image, max_value=255, block_size=7, C=0):
    # Image must be in grayscale
    image_np = np.array(image)

    kernel = np.ones((block_size, block_size)) / (block_size ** 2)
    image_convolved_np = scipy.signal.convolve2d(image_np, kernel, mode='same', boundary='symm')
    image_result_np = image_convolved_np - image_np - C
    image_result = threshold_image(image_result_np, op='>')

    return image_result


# In[10]:


print(gaussian_kernel(3,1))


# In[208]:


for idx, image_name in enumerate(images):
    image = Image.open(os.path.join(image_folder, image_name))
    # print(image.format, image.mode)
    image = image.convert("L")

    with open(os.path.join(text_folder, texts[idx]), 'r') as f:
            base_text = f.readlines()
            base_text = "".join(base_text)
            # base_text = [line.strip() for line in base_text]
    img_cv = numpy.array(image)
    img_th_cv = cv.adaptiveThreshold(img_cv, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,                               cv.THRESH_BINARY, 11, 8)
    
    image_adaptive_gaussian = Image.fromarray(img_th_cv)
    # image_adaptive_gaussian.show()
    result_adaptive_gaussian = pytesseract.image_to_string(image_adaptive_gaussian)
    # print(result_adaptive_gaussian)

    print("Adaptive gaussian:")
    evaluate(result_adaptive_gaussian, base_text)


# ### Self implementation of Adaptive Gaussian thresholding

# In[209]:


for idx, image_name in enumerate(images):
    image = Image.open(os.path.join(image_folder, image_name))
    # print(image.format, image.mode)
    image = image.convert("L")

    with open(os.path.join(text_folder, texts[idx]), 'r') as f:
            base_text = f.readlines()
            base_text = "".join(base_text)
            # base_text = [line.strip() for line in base_text]

    image_th = adaptive_gaussian_thresholding_in(image, block_size=11, std=2, C=8)
    image_th.show()
    result_th = pytesseract.image_to_string(image_th)
    

    evaluate(result_th, base_text)


# In[210]:


# Parameters fine-tuning
accuracy = [0,0]
block_size_optimum = [0,0]
std_optimum = [0,0]
C_optimum = [0,0]
for idx, image_name in enumerate(images):
    image = Image.open(os.path.join(image_folder, image_name))
    # print(image.format, image.mode)
    image = image.convert("L")

    with open(os.path.join(text_folder, texts[idx]), 'r') as f:
            base_text = f.readlines()
            base_text = "".join(base_text)
            # base_text = [line.strip() for line in base_text]
    for C in range(0,10):
        for block_size in range(3,13,2):
            for std in range(1,3):
                image_th = adaptive_gaussian_thresholding_in(image, block_size=block_size,std=std,C=C)
                # image_th.show()
                result_th = pytesseract.image_to_string(image_th)
                score = evaluate(result_th, base_text,False)
                if accuracy[idx] < score:
                    print(f"Found better accuracy of {score} for image {image_name} with parameters {block_size} {std} {C}")
                    accuracy[idx] = score
                    block_size_optimum[idx] = block_size
                    std_optimum[idx] = std
                    C_optimum[idx] = C
                # print(f"{block_size} | {std} | {C} | {score:.5f}")
print(accuracy)
print(block_size_optimum)
print(std_optimum)
print(C_optimum)


# # Gaussian Blur + Adaptive Gaussian Thresholding

# In[9]:


def gaussian_blur_in(image, kernel_size=7, std=1):
    image_np = np.array(image)
    kernel = gaussian_kernel(kernel_size=kernel_size, std=std)
    image_convolved_np = scipy.signal.convolve2d(image_np, kernel, mode='same', boundary='symm')
    return Image.fromarray(image_convolved_np)
    


# In[26]:


for kernel_size in range(3,17,2):
        image = Image.open(os.path.join(image_folder, image_name))
        # print(image.format, image.mode)
        image = image.convert("L")

        with open(os.path.join(text_folder, texts[idx]), 'r') as f:
                base_text = f.readlines()
                base_text = "".join(base_text)
                # base_text = [line.strip() for line in base_text]
        img_cv = numpy.array(image)
        img_blur = cv.GaussianBlur(img_cv, (kernel_size, kernel_size), 0)
#         img_th_cv = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
#                                   cv.THRESH_BINARY, 11, 8)
        image_th = adaptive_gaussian_thresholding_in(img_blur, block_size=9, std=2, C=4)
        # image_th.show()
        result_th = pytesseract.image_to_string(image_th)
        score = evaluate(result_th, base_text, print_score=False)
        print(f"Gaussian blur ({kernel_size},{kernel_size}) + Adaptive gaussian for {image_name} score: {score:.5f}")


# In[29]:


accuracy = [0,0]
kernel_size_optimum = [0,0]
std_optimum = [0,0]
for idx, image_name in enumerate(images):
    for kernel_size in range(3,17,2):
        for std in [0.5,1,2]:
            image = Image.open(os.path.join(image_folder, image_name))
            # print(image.format, image.mode)
            image = image.convert("L")

            with open(os.path.join(text_folder, texts[idx]), 'r') as f:
                    base_text = f.readlines()
                    base_text = "".join(base_text)
                    # base_text = [line.strip() for line in base_text]
            image = gaussian_blur_in(image, kernel_size=kernel_size, std=std)
            image_th = adaptive_gaussian_thresholding_in(image, block_size=15, std=2, C=4)
            # image_th.show()
            result_th = pytesseract.image_to_string(image_th)
            score = evaluate(result_th, base_text, print_score=False)
            if accuracy[idx] < score:
                print(f"Found better accuracy of {score} for image {image_name} with parameters {kernel_size} {std}")
                accuracy[idx] = score
                kernel_size_optimum[idx] = kernel_size
                std_optimum[idx] = std
            # print(f"Gaussian blur ({kernel_size},{kernel_size}) std={std} + Adaptive gaussian for {image_name} score: {score:.5f}")


# In[31]:


print(accuracy)
print(kernel_size_optimum)
print(std_optimum)


# # Additional Testing

# In[13]:


for idx, image_name in enumerate(images):
    image = Image.open(os.path.join(image_folder, image_name))
    # print(image.format, image.mode)
    image = image.convert("L")

    with open(os.path.join(text_folder, texts[idx]), 'r') as f:
            base_text = f.readlines()
            base_text = "".join(base_text)
            # base_text = [line.strip() for line in base_text]
    image = gaussian_blur_in(image, kernel_size=3, std=1)
    image_th = adaptive_gaussian_thresholding_in(image, block_size=9, std=2, C=4)
    image_th.show()
    result_th = pytesseract.image_to_string(image_th)
    score = evaluate(result_th, base_text, print_score=False)
    print(f"Gaussian blur + Adaptive gaussian for {image_name} score: {score:.5f}")
    print(result_th)


# In[21]:


for idx, image_name in enumerate(images):
    if idx==0:
        continue
    for kernel_size in range(3,25,2):
        image = Image.open(os.path.join(image_folder, image_name))
        # print(image.format, image.mode)
        image = image.convert("L")

        with open(os.path.join(text_folder, texts[idx]), 'r') as f:
                base_text = f.readlines()
                base_text = "".join(base_text)
                # base_text = [line.strip() for line in base_text]
        image_cv = np.array(image)
        image_cv = cv.pyrUp(image_cv)
        image = Image.fromarray(image_cv)
        image_th = adaptive_gaussian_thresholding_in(image, block_size=kernel_size, std=2, C=4)
        # image_th.show()
        result_th = pytesseract.image_to_string(image_th)
        score = evaluate(result_th, base_text, print_score=False)
        print(f"Adaptive gaussian {kernel_size} for {image_name} score: {score:.5f}")
        # print(result_th)
        s = difflib.SequenceMatcher(None, result_th, base_text)


# In[ ]:




