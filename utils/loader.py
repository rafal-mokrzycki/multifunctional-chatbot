import repackage
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.openapi.shared.exceptions import (
    NotFoundException,
    PineconeApiException,
)

repackage.up()
from cfg.config import load_config

config = load_config()


class PineconeConnection:
    def __init__(
        self,
        index_name: str | None = None,
        pinecone_api_key: str | None = None,
        metrics: str = "cosine",
        dimension: int = 1536,
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        if index_name is None:
            self.index_name = config["pinecone"]["index_name"]
        else:
            self.index_name = index_name
        if pinecone_api_key is None:
            self.pinecone_api_key = config["pinecone"]["api_key"]
        else:
            self.pinecone_api_key = pinecone_api_key
        self.pinecone = Pinecone(api_key=self.pinecone_api_key)
        self.metrics = metrics
        self.dimension = dimension
        self.cloud = cloud
        self.region = region
        self.embeddings_model = OpenAIEmbeddings(api_key=config["openai"]["api_key"])

    def create_index(self) -> None:
        """
        Creates index if not exists.
        """
        try:
            self.pinecone.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"Index `{self.index_name}` created.")
        except PineconeApiException:
            print(f"Index `{self.index_name}` already exists.")

    def _delete_index(self) -> None:
        """
        Deletes index.
        """
        try:
            self.pinecone.delete_index(self.index_name)
            print(f"Index `{self.index_name}` deleted.")
        except NotFoundException:
            print(f"Index `{self.index_name}` not found.")

    def _delete_data(self, namespace: str) -> None:
        """
        Deletes all data in a given namespace.

        Args:
            namespace (str): Namespace to delete data in.
        """
        self.pinecone.Index(self.index_name).delete(
            delete_all=True, namespace=namespace
        )
        print(f"All data in namespace `{namespace}` successfully deleted.")

    def show_statistics(self):
        """
        Prints out Pinecone index statistics in JSON format.

        Example:
        --------
        {
        'deletion_protection': 'disabled',
        'dimension': 1536,
        'host': 'my-index-828a150.svc.aped-4627-b74a.pinecone.io',
        'metric': 'cosine',
        'name': 'my-index',
        'spec': {
            'serverless': {
                'cloud': 'aws', 'region': 'us-east-1'
            }
        },
        'status': {
            'ready': True, 'state': 'Ready'
            }
        }
        """
        return self.pinecone.Index(self.index_name).describe_index_stats()

    def __repr__(self):
        return str(self.pinecone.describe_index(self.index_name))

    def load_dataset_and_upsert(
        self, dataset: list[str], ids: list[str], namespace: str | None = None
    ) -> None:
        """
        Loads a dataset for embeddings and upserts them into the Pinecone index.

        Args:
            dataset (list[str]): List of text data to be embedded.
            ids (list[str]): List of unique identifiers for each text entry.
            namespace (str): Optional namespace for organizing data.

        Raises:
            ValueError: If the length of dataset and ids do not match.
        """
        if namespace is None:
            namespace = config["pinecone"]["namespace"]["raw"]
        if len(dataset) != len(ids):
            raise ValueError("The length of dataset and ids must match.")

        # Generate embeddings for the dataset
        embeddings = self.embeddings_model.embed_documents(dataset)

        # Prepare vectors for upsert
        vectors_to_upsert = [(ids[i], embeddings[i]) for i in range(len(dataset))]

        # Upsert vectors into Pinecone index
        try:
            self.pinecone.Index(self.index_name).upsert(
                vectors=vectors_to_upsert, namespace=namespace
            )
        except NotFoundException:
            self.create_index()
            self.pinecone.Index(self.index_name).upsert(
                vectors=vectors_to_upsert, namespace=namespace
            )
        print(
            f"Successfully upserted {len(vectors_to_upsert)} vectors \
                into the index `{self.index_name}`."
        )


if __name__ == "__main__":
    dataset = [
        "Your boarding pass may be on your phone or be printed on a piece of paper."
    ]
    ids = ["0"]
    pc = PineconeConnection()
    pc.load_dataset_and_upsert(dataset, ids)
