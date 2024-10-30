"""
This module provides functionality to extract text content from PDF files.
It includes methods to convert PDF pages to image bytes and analyze the content
of these images using a vision model.
"""

from io import BytesIO
from typing import Dict, Any, List

import ollama
from loguru import logger
from pdf2image import convert_from_path


class PDFContentExtractor:
    """
    A class used to retrieve and extract text content from PDF files.
    """

    @staticmethod
    def pdf_to_image_bytes(pdf_path: str) -> List[bytes]:
        """
        Converts each page of a PDF file to an image in byte format.

        Args:
            pdf_path (str): The file path to the PDF document.

        Returns:
            List[bytes]: A list of images in byte format, each representing a
            PDF page.
        """
        pages = convert_from_path(pdf_path)

        page_images = [
            (
                img_byte_arr := BytesIO(),
                page.save(img_byte_arr, format="PNG"),
                img_byte_arr.getvalue(),
            )[2]
            for page in pages
        ]

        return page_images

    @staticmethod
    def get_text_from_image_bytes(page_bytes: bytes) -> str:
        """
        Analyzes a single page image in byte format and extracts its main text
        content.

        Args:
            page_bytes (bytes): The content of a document page in image byte
            format.

        Returns:
            str: The extracted text content from the document page.
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
            "images": [page_bytes],
        }

        response = ollama.chat(
            model="llava:13b",
            messages=[system_message, message],
            options={"temperature": 0},
        )

        logger.info("Response received")

        extracted_text: str = response["message"]["content"]

        return extracted_text
