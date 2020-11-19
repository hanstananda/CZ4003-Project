from pprint import pprint

import cv2 as cv
import numpy
from PIL import Image
import pytesseract
import difflib
import os

image_folder = "./images"
text_folder = "./source"
images = ["sample01.png", "sample02.png"]
texts = ["sample01.txt", "sample02.txt"]


def otsu_tresholding(image_pil):
    img_cv = cv.cvtColor(numpy.array(image_pil), cv.COLOR_RGB2GRAY)
    # Otsu's thresholding
    ret, th = cv.threshold(img_cv, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    img_th_rgb = cv.cvtColor(th, cv.COLOR_GRAY2RGB)
    img_th_pil = Image.fromarray(img_th_rgb)
    # img_th_pil.show()
    return img_th_pil


def apply_median_filter(image_pil):
    img_cv = cv.cvtColor(numpy.array(image_pil), cv.COLOR_RGB2BGR)
    img_blur = cv.medianBlur(img_cv, 3)
    img_cv_rgb = cv.cvtColor(img_blur, cv.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv_rgb)
    img_pil.show()
    return img_pil


def apply_gaussian_blur(image_pil):
    img_cv = cv.cvtColor(numpy.array(image_pil), cv.COLOR_RGB2BGR)
    img_blur = cv.GaussianBlur(img_cv, (5, 5), 0)
    img_cv_rgb = cv.cvtColor(img_blur, cv.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv_rgb)
    return img_pil


def bilateral_filter(image_pil):
    img_cv = cv.cvtColor(numpy.array(image_pil), cv.COLOR_RGB2BGR)
    img_blur = cv.bilateralFilter(img_cv, 9, 100, 100)
    img_cv_rgb = cv.cvtColor(img_blur, cv.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv_rgb)
    return img_pil


def adaptive_gaussian_tresholding(image_pil):
    img_cv = cv.cvtColor(numpy.array(image_pil), cv.COLOR_RGB2GRAY)
    th = cv.adaptiveThreshold(img_cv, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                              cv.THRESH_BINARY, 11, 2)
    img_th_rgb = cv.cvtColor(th, cv.COLOR_GRAY2RGB)
    img_th_pil = Image.fromarray(img_th_rgb)
    # img_th_pil.show()
    return img_th_pil


def evaluate(actual, expected):
    s = difflib.SequenceMatcher(None, actual, expected)
    print(s.ratio())
    print(s.get_matching_blocks())


for idx, image_name in enumerate(images):
    image = Image.open(os.path.join(image_folder, image_name))
    print(image.format, image.mode)
    image = image.convert("RGB")
    result = pytesseract.image_to_string(image)
    # result = result.split("\n")
    # pprint(result)

    with open(os.path.join(text_folder, texts[idx]), 'r') as f:
        base_text = f.readlines()
        base_text = "".join(base_text)
        # base_text = [line.strip() for line in base_text]

    evaluate(result, base_text)

    image_th = otsu_tresholding(image)
    result_th = pytesseract.image_to_string(image_th)

    evaluate(result_th, base_text)

    image_adaptive_gaussian = adaptive_gaussian_tresholding(image)
    image_th_adaptive_gaussian = otsu_tresholding(image_adaptive_gaussian)
    image_median_adaptive_gaussian = apply_median_filter(image_adaptive_gaussian)
    result_adaptive_gaussian = pytesseract.image_to_string(image_adaptive_gaussian)
    result_median_adaptive_gaussian = pytesseract.image_to_string(image_median_adaptive_gaussian)
    result_th_adaptive_gaussian = pytesseract.image_to_string(image_th_adaptive_gaussian)
    # print(result_th_adaptive_gaussian)
    print("Adaptive gaussian:")
    evaluate(result_adaptive_gaussian, base_text)
    print("Adaptive gaussian + median filter:")
    evaluate(result_median_adaptive_gaussian, base_text)
    print("Adaptive gaussian + Otsu threshold:")
    evaluate(result_th_adaptive_gaussian, base_text)

    image_gaussian_blur = apply_gaussian_blur(image)
    image_adaptive_gaussian_blur = adaptive_gaussian_tresholding(image_gaussian_blur)
    image_th_adaptive_gaussian_blur = otsu_tresholding(image_adaptive_gaussian_blur)
    result_th_adaptive_gaussian_blur = pytesseract.image_to_string(image_th_adaptive_gaussian_blur)
    result_adaptive_gaussian_blur = pytesseract.image_to_string(image_adaptive_gaussian_blur)
    print("Adaptive gaussian + gaussian blur: ")
    evaluate(result_adaptive_gaussian_blur, base_text)
    print("Adaptive gaussian + gaussian blur + Otsu threshold: ")
    evaluate(result_th_adaptive_gaussian_blur, base_text)

    image_bilateral_filter = bilateral_filter(image)
    image_adaptive_gaussian_bilateral_filter = adaptive_gaussian_tresholding(image_bilateral_filter)
    image_th_adaptive_gaussian_bilateral_filter = otsu_tresholding(image_adaptive_gaussian_bilateral_filter)
    result_adaptive_gaussian_bilateral_filter = pytesseract.image_to_string(image_adaptive_gaussian_bilateral_filter)
    result_th_adaptive_gaussian_bilateral_filter = pytesseract.image_to_string(image_th_adaptive_gaussian_bilateral_filter)
    print("Adaptive gaussian + bilateral filter: ")
    evaluate(result_adaptive_gaussian_bilateral_filter, base_text)

    print("Adaptive gaussian + bilateral filter + Otsu threshold: ")
    evaluate(result_th_adaptive_gaussian_bilateral_filter, base_text)

    break

