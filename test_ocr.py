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
    if image.mode == "LA":
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

    image_th_adaptive_gaussian = adaptive_gaussian_tresholding(image)
    image_th_adaptive_gaussian = otsu_tresholding(image_th_adaptive_gaussian)
    result_th_adaptive_gaussian = pytesseract.image_to_string(image_th_adaptive_gaussian)
    print(result_th_adaptive_gaussian)
    evaluate(result_th_adaptive_gaussian, base_text)

    break
