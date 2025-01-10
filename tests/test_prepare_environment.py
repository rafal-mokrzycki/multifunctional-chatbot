import os
from unittest.mock import MagicMock, patch

import pytest
import repackage

repackage.up()
from utils.constants import ConstantsManagement
from utils.prepare_environment import PDFProcessor, PDFProcessorBuilder

CONSTANTS = ConstantsManagement()


@pytest.fixture
def pdf_processor():
    """Fixture to create a PDFProcessor instance."""
    return PDFProcessor()


def test_load_pdf(pdf_processor):
    """Test loading PDF files."""
    with patch("os.listdir", return_value=["test.pdf"]), patch(
        "langchain_community.document_loaders.PyPDFLoader.load",
        return_value=[MagicMock(page_content="Sample content.")],
    ):
        pdf_processor.load_pdf()
        assert pdf_processor.text == "Sample content."


def test_clean_text_with_replacement(pdf_processor):
    """Test cleaning text with replacement dictionary."""
    pdf_processor.text = "Sample 08 text. For more information, please visit"
    replacement_dict = {"Sample": "Example"}

    pdf_processor.clean_text(replacement_dict)

    assert pdf_processor.text == "Example text. "


def test_clean_text_without_replacement(pdf_processor):
    """Test cleaning text without a replacement dictionary."""
    pdf_processor.text = "Sample 08 text. For more information, please visit."

    with patch.object(
        pdf_processor, "_remove_numbers", return_value="text."
    ) as mock_remove_numbers, patch.object(
        pdf_processor, "_remove_footer", return_value="text."
    ):
        pdf_processor.clean_text()

        mock_remove_numbers.assert_called_once()
        mock_remove_numbers.assert_called_once()
        assert pdf_processor.text == "text."


def test_chunk_text(pdf_processor):
    """Test chunking text."""
    pdf_processor.text = "This is a sample text for chunking."

    with patch(
        "langchain_experimental.text_splitter.SemanticChunker.split_text",
        return_value=["This is a sample", "text for chunking."],
    ):
        chunks = pdf_processor.chunk_text()

        assert chunks == ["This is a sample", "text for chunking."]


def test_generate_embeddings(pdf_processor):
    """Test generating embeddings from chunks."""
    chunks = ["This is a sample", "text for chunking."]

    # Mock the tokenizer and OpenAI client response
    with patch(
        "tiktoken.get_encoding",
        return_value=MagicMock(encode=MagicMock(return_value=[1])),
    ), patch(
        "openai.OpenAI.embeddings.create",
        return_value=MagicMock(data=[MagicMock(embedding=[0.1] * 1536)]),
    ):

        pdf_processor.generate_embeddings(chunks)

        assert (
            len(pdf_processor.embeddings) == 2
        )  # Two chunks should yield two embeddings


def test_upsert_embeddings(pdf_processor):
    """Test upserting embeddings into Pinecone."""

    # Mocking Pinecone and environment setup
    with patch("pinecone.Pinecone") as mock_pinecone:
        mock_index = MagicMock()
        mock_pinecone.return_value.Index.return_value = mock_index

        # Mock the embeddings and chunking process
        pdf_processor.embeddings = [[0.1] * 1536]
        pdf_processor.text = ["This is a sample"]

        # Mock list_indexes to simulate existing index
        mock_pinecone.return_value.list_indexes.return_value = [
            CONSTANTS.PINECONE_INDEX_NAME
        ]

        # Run upsert_embeddings
        pdf_processor.upsert_embeddings()

        # Check if upsert was called correctly
        mock_index.upsert.assert_called_once()


if __name__ == "__main__":
    pytest.main()
