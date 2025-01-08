import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import repackage

repackage.up()
from utils.preprocessor import (
    chunk_text,
    generate_embeddings,
    get_chunks,
    get_dataset_from_text,
    load_pdf,
    prepare_dataset,
    replace_forbidden_chars,
    tiktoken_len,
)


@pytest.fixture
def sample_text():
    return """This is a sample text for testing purposes.
    This is a sample text for testing purposes.
    This is a sample text for testing purposes.
    This is a sample text for testing purposes."""


@pytest.fixture
def sample_embeddings():
    return [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]


def test_tiktoken_len(sample_text):
    length = tiktoken_len(sample_text)
    assert length > 0


def test_get_chunks(sample_text):
    chunks = get_chunks(sample_text)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


@patch("client.embeddings.create")
def test_generate_embeddings(mock_create, sample_embeddings):
    mock_create.return_value.data = [{"embedding": [0.1] * 1536}]

    embeddings = generate_embeddings(["sample text"])

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 1536


@patch("utils.preprocessor.load_pdf")
def test_prepare_dataset(mock_load_pdf, sample_embeddings):
    mock_load_pdf.return_value = "Sample PDF content."

    with patch(
        "utils.preprocessor.replace_forbidden_chars", return_value="Cleaned text"
    ):
        with patch("utils.preprocessor.chunk_text", return_value=["Chunked text"]):
            with patch(
                "utils.preprocessor.generate_embeddings", return_value=sample_embeddings
            ):
                dataset = prepare_dataset()
                assert isinstance(dataset, pd.DataFrame)
                assert "id" in dataset.columns
                assert "values" in dataset.columns


def test_replace_forbidden_chars():
    text = "This • is a sample\ntext."
    replaced = replace_forbidden_chars(text)
    assert replaced == "This is a sample text."


def test_chunk_text(sample_text):
    chunks = chunk_text(sample_text)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


@patch("os.listdir")
@patch("langchain_community.document_loaders.PyPDFLoader")
def test_load_pdf(mock_loader, mock_listdir):
    mock_listdir.return_value = ["test.pdf"]

    # Mocking the loader's load method to return a list of documents with page content
    mock_loader.return_value.load.return_value = [
        MagicMock(page_content="Mocked Page content")
    ]

    content = load_pdf()

    assert "Mocked Page content" in content


def test_get_dataset_from_text(sample_embeddings):
    df = get_dataset_from_text(sample_embeddings)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == len(
        sample_embeddings
    )  # Number of rows should match embeddings
