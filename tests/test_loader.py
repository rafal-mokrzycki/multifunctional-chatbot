from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import repackage

repackage.up()
from utils.loader import load_data_from_dataset


@pytest.fixture
def mock_pinecone():
    """Fixture to mock the Pinecone instance."""
    with patch("pinecone.Pinecone") as mock:
        yield mock


@pytest.fixture
def mock_prepare_dataset():
    """Fixture to mock prepare_dataset function."""
    with patch("pinecone.prepare_dataset") as mock:
        yield mock


@pytest.fixture
def sample_dataframe():
    """Fixture for a sample DataFrame."""
    return pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "values": [
                [0.2342352] * 1536,
                [0.34654542] * 1536,
                [0.877564] * 1536,
            ],
        }
    )


def test_load_data_with_provided_dataframe(mock_pinecone, sample_dataframe):
    # Arrange
    mock_pinecone_instance = mock_pinecone.return_value

    # Ensure list_indexes returns an empty list to trigger index creation
    mock_pinecone_instance.list_indexes.return_value.names.return_value = []
    mock_pinecone_instance.describe_index.return_value.status = {"ready": True}

    # Mocking the Index class and its upsert method
    mock_index = MagicMock()
    mock_pinecone_instance.Index.return_value = mock_index

    # Act
    load_data_from_dataset(sample_dataframe)

    # Assert
    mock_pinecone_instance.create_index.assert_called_once()
    mock_index.upsert_from_dataframe.assert_called_once_with(
        sample_dataframe, batch_size=100
    )


def test_load_data_without_provided_dataframe(mock_pinecone, mock_prepare_dataset):
    # Arrange
    mock_prepare_dataset.return_value = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "values": [
                [0.2342352, 0.14325323, 0.2343452],
                [0.34654542, 0.356253, 0.637452],
                [0.877564, 0.7896543, 0.7986543],
            ],
        }
    )
    mock_pinecone_instance = mock_pinecone.return_value
    mock_pinecone_instance.list_indexes.return_value.names.return_value = []
    mock_pinecone_instance.describe_index.return_value.status = {"ready": True}

    # Act
    load_data_from_dataset()

    # Assert
    mock_prepare_dataset.assert_called_once()
    assert mock_pinecone_instance.create_index.called
    assert mock_pinecone_instance.Index().upsert_from_dataframe.called


def test_load_data_index_exists(mock_pinecone, sample_dataframe):
    # Arrange
    mock_pinecone_instance = mock_pinecone.return_value
    mock_pinecone_instance.list_indexes.return_value.names.return_value = [
        "existing_index"
    ]

    # Act
    load_data_from_dataset(sample_dataframe)

    # Assert
    assert not mock_pinecone_instance.create_index.called
