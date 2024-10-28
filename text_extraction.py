import glob
import os
from io import BytesIO

import ollama
from loguru import logger
from pdf2image import convert_from_path
from pyprojroot import here

if __name__ == "__main__":
    pdf_path = str(here("./data/"))

    pdf_files = glob.glob(os.path.join(pdf_path, "*.pdf"))

    pages = convert_from_path(pdf_files[0])

    images_bytes = [
        (
            img_byte_arr := BytesIO(),
            page.save(img_byte_arr, format="PNG"),
            img_byte_arr.getvalue(),
        )[2]
        for page in pages
    ]

    message = {
        "role": "user",
        "content": "Describe this image:",
        "images": [images_bytes[0]],
    }

    response = ollama.chat(model="llava:13b", messages=[message])

    logger.info("Response received")

    logger.info(response["message"]["content"])
