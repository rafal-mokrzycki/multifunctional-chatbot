import pytest
from pinecone.core.openapi.shared.exceptions import (
    NotFoundException,
    PineconeApiException,
)

from utils.loader import PineconeConnection


@pytest.fixture
def pinecone_connection(mocker):
    mock_pinecone = mocker.patch("utils.loader.Pinecone")  # Mock Pinecone class
    mock_embeddings_model = mocker.patch(
        "langchain_openai.OpenAIEmbeddings"
    )  # Mock OpenAIEmbeddings class
    return PineconeConnection(index_name="test-index", pinecone_api_key="test-api-key")


def test_create_index(pinecone_connection, mocker):
    mocker.patch.object(pinecone_connection.pinecone, "create_index", return_value=None)
    pinecone_connection.create_index()
    pinecone_connection.pinecone.create_index.assert_called_once_with(
        name="test-index",
        dimension=1536,
        metric="cosine",
        spec=mocker.ANY,  # You can further specify if needed
    )


def test_create_index_already_exists(pinecone_connection, mocker):
    mocker.patch.object(
        pinecone_connection.pinecone, "create_index", side_effect=PineconeApiException
    )
    pinecone_connection.create_index()


def test_delete_index(pinecone_connection, mocker):
    mocker.patch.object(pinecone_connection.pinecone, "delete_index", return_value=None)
    pinecone_connection._delete_index()
    pinecone_connection.pinecone.delete_index.assert_called_once_with("test-index")


def test_delete_index_not_found(pinecone_connection, mocker):
    mocker.patch.object(
        pinecone_connection.pinecone, "delete_index", side_effect=NotFoundException
    )
    pinecone_connection._delete_index()


def test_delete_data(pinecone_connection, mocker):
    mock_delete = mocker.patch.object(
        pinecone_connection.pinecone.Index("test-index"), "delete", return_value=None
    )
    pinecone_connection._delete_data(namespace="test-namespace")
    mock_delete.assert_called_once_with(delete_all=True, namespace="test-namespace")


def test_show_statistics(pinecone_connection, mocker):
    expected_stats = {"status": {"ready": True}}
    mocker.patch.object(
        pinecone_connection.pinecone.Index("test-index"),
        "describe_index_stats",
        return_value=expected_stats,
    )
    stats = pinecone_connection.show_statistics()
    assert stats == expected_stats


def test_load_dataset_and_upsert(pinecone_connection, mocker):
    dataset = ["sample text"]
    ids = ["0"]

    # Mock embedding generation
    embeddings_mock = [0.1] * 1536  
    mocker.patch.object(
        pinecone_connection.embeddings_model,
        "embed_documents",
        return_value=[embeddings_mock],
    )

    # Mock upsert operation
    upsert_mock = mocker.patch.object(
        pinecone_connection.pinecone.Index("test-index"), "upsert", return_value=None
    )

    pinecone_connection.load_dataset_and_upsert(dataset, ids)

    upsert_mock.assert_called_once()


# @pytest.fixture(name="pinecone_index_name")
# def pinecone_index_name():
#     yield "british-airways-test"


# def test_create_index(pinecone_index_name):
#     pi = PineconeConnection(index_name=pinecone_index_name)
#     pi.create_index()
#     assert pinecone_index_name in pi.pinecone.list_indexes().names()
#     pi.pinecone.delete_index(pinecone_index_name)


# def test_delete_index(pinecone_index_name):
#     pi = PineconeConnection(index_name=pinecone_index_name)
#     pi.create_index()
#     pi._delete_index()
#     assert pinecone_index_name not in pi.pinecone.list_indexes().names()


# def test_delete_data(pinecone_index_name):
#     # Sample data
#     dataset = ["Hello", "World"]
#     ids = [1, 2]
#     pi = PineconeConnection(index_name=pinecone_index_name)
#     pi.create_index()
#     pi.load_dataset_and_upsert(dataset=dataset, ids=ids, namespace="sentences_raw")
#     pi._delete_data(namespace="sentences_raw")
#     description = pi.pinecone.Index(pinecone_index_name).describe_index_stats()
#     assert description["total_vector_count"] == 0
#     pi.pinecone.delete_index(pinecone_index_name)


# def test_load_dataset_and_upsert(pinecone_index_name):
#     dataset = ["Hello", "World"]
#     ids = [1, 2]
#     pi = PineconeConnection(index_name=pinecone_index_name)
#     pi.create_index()
#     pi.load_dataset_and_upsert(dataset=dataset, ids=ids, namespace="sentences_raw")
#     description = pi.pinecone.Index(pinecone_index_name).describe_index_stats()
#     assert description["total_vector_count"] == 2
#     pi.pinecone.delete_index(pinecone_index_name)
