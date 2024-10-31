from unittest.mock import patch, MagicMock

import pytest

from src.text_extraction import PDFContentExtractor


@pytest.fixture
def mock_pdf_page_bytes():
    # A simple byte representation of a PNG image, for testing
    return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00"


@pytest.fixture
def mock_extracted_text():
    # Sample text returned from the Ollama API mock
    return "This is a mock extracted text from the PDF page."


class TestPdfToImageBytes:
    @patch("src.text_extraction.convert_from_path")
    def test_empty_pdf(self, mock_convert_from_path):
        """
        Test the behavior when an empty PDF is provided.
        Should return an empty list if there are no pages.
        """
        mock_convert_from_path.return_value = []

        pdf_path = "empty_test.pdf"
        result = PDFContentExtractor.pdf_to_image_bytes(pdf_path)

        # Expected result is an empty list
        assert result == []
        mock_convert_from_path.assert_called_once_with(pdf_path)

    @patch("src.text_extraction.convert_from_path")
    @patch("src.text_extraction.BytesIO")
    def test_single_page(
        self, mock_bytesio, mock_convert_from_path, mock_pdf_page_bytes
    ):
        """
        Test the behavior when a PDF with a single page is provided.
        Should return a list with a single byte object.
        """
        # Mock a single page image conversion to bytes
        mock_image = MagicMock()
        mock_image.save.side_effect = lambda *args, **kwargs: None
        mock_convert_from_path.return_value = [mock_image]

        # Prepare BytesIO mock to capture saved image data
        mock_img_byte_arr = MagicMock()
        mock_bytesio.return_value = mock_img_byte_arr
        mock_img_byte_arr.getvalue.return_value = mock_pdf_page_bytes

        pdf_path = "single_page.pdf"
        result = PDFContentExtractor.pdf_to_image_bytes(pdf_path)

        # Check the result for a single page conversion
        assert result == [mock_pdf_page_bytes]
        mock_convert_from_path.assert_called_once_with(pdf_path)
        assert mock_image.save.called
        assert mock_img_byte_arr.getvalue.called

    @patch("src.text_extraction.convert_from_path")
    @patch("src.text_extraction.BytesIO")
    def test_multiple_pages(
        self, mock_bytesio, mock_convert_from_path, mock_pdf_page_bytes
    ):
        """
        Test the behavior when a PDF with multiple pages is provided.
        Should return a list with a byte object for each page.
        """
        # Mock multiple pages (e.g., 3 pages) image conversion to bytes
        mock_image1, mock_image2, mock_image3 = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_image1.save.side_effect = lambda *args, **kwargs: None
        mock_image2.save.side_effect = lambda *args, **kwargs: None
        mock_image3.save.side_effect = lambda *args, **kwargs: None
        mock_convert_from_path.return_value = [
            mock_image1,
            mock_image2,
            mock_image3,
        ]

        # Prepare BytesIO mock to capture saved image data
        mock_img_byte_arr = MagicMock()
        mock_bytesio.return_value = mock_img_byte_arr
        mock_img_byte_arr.getvalue.return_value = mock_pdf_page_bytes

        pdf_path = "multiple_pages.pdf"
        result = PDFContentExtractor.pdf_to_image_bytes(pdf_path)

        # Expected result is a list with a byte object for each page
        assert result == [
            mock_pdf_page_bytes,
            mock_pdf_page_bytes,
            mock_pdf_page_bytes,
        ]
        mock_convert_from_path.assert_called_once_with(pdf_path)
        assert mock_image1.save.called
        assert mock_image2.save.called
        assert mock_image3.save.called
        assert mock_img_byte_arr.getvalue.called

    @patch("src.text_extraction.convert_from_path")
    def test_no_pdf_provided(self, mock_convert_from_path):
        """
        Test the behavior when no PDF is provided.
        Should raise a TypeError or appropriate exception.
        """
        mock_convert_from_path.side_effect = TypeError("No file path provided")

        with pytest.raises(TypeError, match="No file path provided"):
            PDFContentExtractor.pdf_to_image_bytes(None)

        mock_convert_from_path.assert_called_once_with(None)


class TestGetTextFromImageBytes:
    @patch("src.text_extraction.ollama.chat")
    def test_content(
        self, mock_ollama_chat, mock_pdf_page_bytes, mock_extracted_text
    ):
        """
        Test that content is correctly extracted from the image bytes
        when the Ollama API returns valid text content.
        """
        # Mock the response from the Ollama API with actual content
        mock_response = {"message": {"content": mock_extracted_text}}
        mock_ollama_chat.return_value = mock_response

        # Run the method with mocked dependencies
        result = PDFContentExtractor.get_text_from_image_bytes(
            mock_pdf_page_bytes
        )

        # Check if the Ollama API was called with correct parameters
        mock_ollama_chat.assert_called_once()
        args, kwargs = mock_ollama_chat.call_args
        assert kwargs["model"] == "llava:13b"
        assert kwargs["options"] == {"temperature": 0}
        assert "messages" in kwargs
        assert kwargs["messages"][1]["images"] == [mock_pdf_page_bytes]

        # Verify the result matches the expected extracted text
        assert result == mock_extracted_text

    @patch("src.text_extraction.ollama.chat")
    def test_no_content(self, mock_ollama_chat, mock_pdf_page_bytes):
        """
        Test the behavior when the Ollama API returns no content.
        The method should return an empty string in this case.
        """
        # Mock the response from the Ollama API with an empty content string
        mock_response = {"message": {"content": ""}}
        mock_ollama_chat.return_value = mock_response

        # Run the method with mocked dependencies
        result = PDFContentExtractor.get_text_from_image_bytes(
            mock_pdf_page_bytes
        )

        # Verify that the method handles empty content properly
        assert result == ""
        mock_ollama_chat.assert_called_once()

        # Verify that the method called Ollama API with the expected parameters
        args, kwargs = mock_ollama_chat.call_args
        assert kwargs["model"] == "llava:13b"
        assert kwargs["options"] == {"temperature": 0}
        assert "messages" in kwargs
        assert kwargs["messages"][1]["images"] == [mock_pdf_page_bytes]
