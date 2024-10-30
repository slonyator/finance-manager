"""
This module serves as the entry point for executing the PDF content extraction process.
It utilizes the PDFContentExtractor class to convert PDF pages to image bytes and
extracts text content from these images.
"""

import glob
import os
from loguru import logger
from pyprojroot import here
from text_extraction import PDFContentExtractor

if __name__ == "__main__":
    DATA_DIRECTORY = str(here("./data/"))

    pdf_files = glob.glob(os.path.join(DATA_DIRECTORY, "*.pdf"))

    image_bytes = PDFContentExtractor.pdf_to_image_bytes(pdf_files[0])

    content = PDFContentExtractor.get_text_from_image_bytes(image_bytes[0])

    logger.info(f"Extracted Content: {content}")
