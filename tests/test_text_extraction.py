import pytest
from unittest.mock import patch, MagicMock
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
    @patch("src.text_extraction.BytesIO")
    def test_pdf_to_image_bytes(
        self, mock_bytesio, mock_convert_from_path, mock_pdf_page_bytes
    ):
        # Mock a single page as an image and simulate its conversion to bytes
        mock_image = MagicMock()
        mock_image.save.side_effect = (
            lambda *args, **kwargs: mock_pdf_page_bytes
        )
        mock_convert_from_path.return_value = [mock_image]

        # Prepare the BytesIO mock to capture saved image data
        mock_img_byte_arr = MagicMock()
        mock_bytesio.return_value = mock_img_byte_arr
        mock_img_byte_arr.getvalue.return_value = mock_pdf_page_bytes

        pdf_path = "test.pdf"
        result = PDFContentExtractor.pdf_to_image_bytes(pdf_path)

        # Verify that convert_from_path was called once with the provided PDF path
        mock_convert_from_path.assert_called_once_with(pdf_path)

        # Check if each page image was converted to bytes correctly
        assert result == [mock_pdf_page_bytes]
        assert mock_image.save.called
        assert mock_img_byte_arr.getvalue.called

    @patch("src.text_extraction.convert_from_path")
    def test_pdf_to_image_bytes_empty_pdf(self, mock_convert_from_path):
        # Test the behavior when there are no pages in the PDF
        mock_convert_from_path.return_value = []

        pdf_path = "empty_test.pdf"
        result = PDFContentExtractor.pdf_to_image_bytes(pdf_path)

        # The result should be an empty list if no pages are found
        assert result == []
        mock_convert_from_path.assert_called_once_with(pdf_path)


class TestGetTextFromImageBytes:
    @patch("src.text_extraction.ollama.chat")
    def test_get_text_from_image_bytes(
        self, mock_ollama_chat, mock_pdf_page_bytes, mock_extracted_text
    ):
        # Mock the response from the Ollama API
        mock_response = {"message": {"content": mock_extracted_text}}
        mock_ollama_chat.return_value = mock_response

        result = PDFContentExtractor.get_text_from_image_bytes(
            mock_pdf_page_bytes
        )

        # Verify that ollama.chat was called with correct parameters
        mock_ollama_chat.assert_called_once()
        args, kwargs = mock_ollama_chat.call_args
        assert kwargs["model"] == "llava:13b"
        assert kwargs["options"] == {"temperature": 0}
        assert "messages" in kwargs
        assert kwargs["messages"][1]["images"] == [mock_pdf_page_bytes]

        # Check if the extracted text is as expected
        assert result == mock_extracted_text

    @patch("src.text_extraction.ollama.chat")
    def test_get_text_from_image_bytes_no_content(
        self, mock_ollama_chat, mock_pdf_page_bytes
    ):
        # Test for a case where no content is returned
        mock_response = {"message": {"content": ""}}
        mock_ollama_chat.return_value = mock_response

        result = PDFContentExtractor.get_text_from_image_bytes(
            mock_pdf_page_bytes
        )

        # Ensure that the method handles empty content properly
        assert result == ""
        mock_ollama_chat.assert_called_once()
