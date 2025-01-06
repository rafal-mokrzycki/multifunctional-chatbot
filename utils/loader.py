import time

import pandas as pd
import repackage
from pinecone import Pinecone, ServerlessSpec

repackage.up()
from cfg.config import load_config

config = load_config()


def load_data_from_dataset(dataset: pd.DataFrame):
    api_key = config["pinecone"]["api_key"]
    index_name = config["pinecone"]["index_name"]
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
