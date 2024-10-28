import glob
import os
from io import BytesIO
from typing import List

import ollama
from loguru import logger
from pdf2image import convert_from_path
from pyprojroot import here


class ContentExtractor:

    def convert_pdf_to_images(self, pdf_path: str) -> List[bytes]:
        """
        Converts a PDF file to a list of images in byte format.

        Args:
            pdf_path (str): The file path to the PDF document.

        Returns:
            List[bytes]: A list of images in byte format.
        """
        pages = convert_from_path(pdf_path)

        images_bytes = [
            (
                img_byte_arr := BytesIO(),
                page.save(img_byte_arr, format="PNG"),
                img_byte_arr.getvalue(),
            )[2]
            for page in pages
        ]

        return images_bytes


if __name__ == "__main__":
    pdf_path = str(here("./data/"))

    pdf_files = glob.glob(os.path.join(pdf_path, "*.pdf"))

    content = ContentExtractor().convert_pdf_to_images(pdf_files[0])

    message = {
        "role": "user",
        "content": "Describe this image:",
        "images": [content[0]],
    }

    response = ollama.chat(model="llava:13b", messages=[message])

    logger.info("Response received")

    logger.info(response["message"]["content"])
