import time

import pandas as pd
import repackage
from pinecone import Pinecone, ServerlessSpec

repackage.up()
from cfg.config import load_config
from utils.preprocessor import prepare_dataset

config = load_config()


def load_data_from_dataset(dataset: pd.DataFrame | None = None) -> None:
    """Reads a dataframe and upserts vectors to vectorstore.

    Args:
        dataset (pd.DataFrame | None, optional): Dataframe to read. If None calls
        a function prepare_dataset() in order to preprocess default files.
        Defaults to None.
    """
    api_key = config["pinecone"]["api_key"]
    index_name = config["pinecone"]["index_name"]
    if dataset is None:
        dataset = prepare_dataset()
    # create Pinecone instance
    pc = Pinecone(api_key=api_key)
    # create index if not exists
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Index `{index_name}` created.")
    else:
        print(f"Index `{index_name}` already exists.")
    # wait for index to be initialized
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    # upsert the data to Pinecone
    index = pc.Index(index_name)
    index.upsert_from_dataframe(dataset, batch_size=100)


if __name__ == "__main__":
    load_data_from_dataset()
