from pprint import pprint

from PIL import Image
import pytesseract
import difflib
import os

image_folder = "./images"
text_folder = "./source"
images = ["sample01.png", "sample02.png"]
texts = ["sample01.txt", "sample02.txt"]

for idx, image_name in enumerate(images):
    image = Image.open(os.path.join(image_folder, image_name))
    print(image.format, image.mode)
    if image.mode == "LA":
        image = image.convert("RGB")
    result = pytesseract.image_to_string(image)
    # result = result.split("\n")
    pprint(result)

    with open(os.path.join(text_folder, texts[idx]), 'r') as f:
        base_text = f.readlines()
        base_text = "".join(base_text)
        # base_text = [line.strip() for line in base_text]

    pprint(base_text)
    s = difflib.SequenceMatcher(None, result, base_text)
    print(s.ratio())
    print(s.get_matching_blocks())

