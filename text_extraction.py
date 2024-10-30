import glob
import os
from io import BytesIO
from typing import Dict, Any
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

    def page_content_to_str(self, page: bytes) -> str:
        """
        Analyzes a document page and extracts the main content.

        Args:
            page (bytes): The content of the document page in byte format.

        Returns:
            str: The extracted content from the document page.
        """
        system_message: Dict[str, Any] = {
            "role": "system",
            "content": (
                "You are a document analysis assistant specialized in "
                "extracting structured information from scanned pages of "
                "documents. Focus on identifying text, headings, tables, and "
                "visual elements. When extracting content, preserve any "
                "structured elements such as bullet points, lists, and tables, "
                "and provide content in a clean, readable format for ease of "
                "understanding."
            ),
        }

        message: Dict[str, Any] = {
            "role": "user",
            "content": (
                "Analyze this document page and extract the main content. "
                "Focus on text, headings, tables, and important visual "
                "elements. Provide a detailed summary that captures the and "
                "key information while preserving any structured elements like "
                "lists or bullet points."
            ),
            "images": [page],
        }

        response = ollama.chat(
            model="llava:13b", messages=[system_message, message]
        )

        logger.info("Response received")

        extracted_content: str = response["message"]["content"]

        logger.info(f"Content: {extracted_content}")

        return extracted_content


if __name__ == "__main__":
    pdf_path = str(here("./data/"))

    pdf_files = glob.glob(os.path.join(pdf_path, "*.pdf"))

    bytes_content = ContentExtractor().convert_pdf_to_images(pdf_files[0])

    content = ContentExtractor().page_content_to_str(bytes_content[0])

    logger.info(f"Content: {content}")
