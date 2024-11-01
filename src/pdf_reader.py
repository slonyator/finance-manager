import ollama

from typing import List, Dict, Any
from PyPDF2 import PdfReader
from loguru import logger

import pdfplumber
import pandas as pd
from pdfplumber.utils import extract_text, get_bbox_overlap, obj_to_bbox


def process_pdf(pdf_path):
    pdf = pdfplumber.open(pdf_path)
    all_text = []

    for page in pdf.pages:
        filtered_page = page
        chars = filtered_page.chars

        for table in page.find_tables():
            first_table_char = page.crop(table.bbox).chars[0]
            filtered_page = filtered_page.filter(
                lambda obj: get_bbox_overlap(obj_to_bbox(obj), table.bbox)
                is None
            )
            chars = filtered_page.chars

            df = pd.DataFrame(table.extract())
            df.columns = df.iloc[0]
            markdown = df.drop(0).to_markdown(index=False)

            chars.append(first_table_char | {"text": markdown})

        page_text = extract_text(chars, layout=True)
        all_text.append(page_text)

    pdf.close()
    return "\n".join(all_text)


def pdf_read(pdf_doc: str) -> List[str]:
    """Read the text from each page of a PDF document.

    Args:
        pdf_doc (str): The file path to the PDF document.

    Returns:
        List[str]: A list containing the text of each page.
    """
    pdf_reader = PdfReader(pdf_doc)
    pages_text = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages_text.append(page_text)
    return pages_text


def restore_pdf_content(page: str, model_name: str = "llava:13b") -> str:

    system_message: Dict[str, Any] = {
        "role": "system",
        "content": (
            "You are a document restoration assistant tasked with restoring text from "
            "OCR-extracted financial tables. Each transaction should be structured in a "
            "table with the following columns: Buchung | Valuta | Vorgang | Soll | Haben.\n\n"
            "Formatting guidelines:\n"
            "- **Table Structure**: Interpret line breaks, spaces, and delimiters to recreate "
            "a clean, readable table structure.\n"
            "- **Column Alignment**: Ensure consistent alignment within each column for "
            "dates, transaction details, and amounts.\n"
            "- **Multi-line Cells**: Retain line breaks in 'Vorgang' when it contains multiple lines.\n"
            "- **Special Characters**: Include symbols like '+', commas, and periods "
            "for monetary values.\n"
            "- **Unclear Text**: Mark ambiguous OCR text with '[unclear]'.\n\n"
            "Deliver output as plain text or Markdown, ensuring tables are clear and well-aligned."
        ),
    }

    message: Dict[str, Any] = {
        "role": "user",
        "content": (
            f"Restore this OCR-extracted text, focusing on accurate tables and "
            f"alignment. Extracted text: {page}"
        ),
    }

    response = ollama.chat(
        model=model_name,
        messages=[system_message, message],
        options={"temperature": 0},
    )

    logger.info("Response received")

    text: str = response["message"]["content"]

    return text


if __name__ == "__main__":
    file = "/Users/michael/Coding/pycologne/src/Postbank-Kontoauszug-Geschaeftskunden-Muster.pdf"
    texts = pdf_read(file)
    print("Original text:")
    print(texts[0])
    restored_text = restore_pdf_content(
        page=texts[0], model_name="llama3.2:latest"
    )
    print("Restored text:")
    print(restored_text)

    print("Another test:")
    print(process_pdf(file))
