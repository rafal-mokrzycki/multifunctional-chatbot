import os

import pandas as pd
import repackage
import spacy
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI

repackage.up()
from cfg.config import load_config

STRINGS_TO_REPLACE = {"• ": "", "\n": " ", "  ": " ", "   ": " "}
TOKENIZER = tiktoken.get_encoding("cl100k_base")
config = load_config()

client = OpenAI(api_key=config["openai"]["api_key"])


def prepare_dataset():
    text = load_pdf()
    text_cleaned = replace_forbidden_chars(text)
    text_chunked = chunk_text(text_cleaned)
    embeddings = generate_embeddings(text_chunked)
    return get_dataset_from_text(embeddings)


def tiktoken_len(text: str) -> int:
    """Returns the length of tokens.

    Args:
        text (str): Text to apply function on.

    Returns:
        int: Length of tokens.
    """
    tokens = TOKENIZER.encode(text, disallowed_special=())
    return len(tokens)


def get_chunks(text: str) -> list[str]:
    """Splits text and returns chunks.

    Args:
        text (str): Text to split.

    Returns:
        list[str]: Chunks after splitting.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""],
    )

    return text_splitter.split_text(text)


def generate_embeddings(text_chunks: list[str]) -> list[list[float]]:
    """Generates embeddingsfor vector store.

    Args:
        text_chunks (list[str]): Text chunks.

    Returns:
        list[list[float]]: Embeddings.
    """
    embeddings = []
    for text in text_chunks:
        if isinstance(text, str):
            inputs = TOKENIZER.encode(text)
            embedding_response = client.embeddings.create(
                input=inputs, model="text-embedding-3-small"
            )
            embedding = embedding_response.data[0].embedding
            embeddings.append(embedding)
        else:
            print(f"Expected string but got {type(text)}: {text}")
    return embeddings


def load_pdf() -> str:
    """Reads PDF files from directory `data` and joins them into one string.

    Returns:
        str: Joined PDF files as string.
    """
    files = []
    for file_name in os.listdir("data"):
        if file_name.endswith(".pdf"):
            filepath = os.path.abspath(os.path.join("data", file_name))
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            pages = []
            for doc in documents:
                pages.append(doc.page_content)
        files.append(" ".join(pages))
    return " ".join(files)


def replace_forbidden_chars(
    text: str | list[str],
    replacement_dict: dict | None = None,
) -> str | list[str]:
    """
    Replaces characters that prevent from reading CSV file into pandas DataFrame.

    Args:
        text (str | list[str]): Text or list of texts to apply replacement on.
        replacement_dict (dict): Dictionary with characters to be replaced (keys) and
        characters to replace with (values). If None, taken from config.json.
        Defaults to None.
    Returns:
        (str | list[str]): Text or list of texts.
    """
    if replacement_dict is None:
        replacement_dict = STRINGS_TO_REPLACE
    elif not isinstance(replacement_dict, dict):
        raise TypeError("Must be a dictionary.")
    if isinstance(text, str):
        for key in replacement_dict:
            text = text.replace(key, replacement_dict[key])
        return text
    elif isinstance(text, list):
        result = []
        for elem in text:
            if not isinstance(elem, str):
                raise TypeError("Text must be string or list of strings.")
            for key in replacement_dict:
                elem = elem.replace(key, replacement_dict[key])
            result.append(elem)
        return result
    else:
        raise TypeError("Text must be string or list of strings.")


def chunk_text(text: str) -> list[str]:
    """
    Chunks text into overlapping 3-sentence chunks (they overlap 1 sentence with previous
    chunk, 1 with next and 1 sentence is non-overlapping.)

    Args:
        text (str): String to chunk.

    Returns:
        list[str]: List of strings after chunking.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    result = []
    for i, _ in enumerate(sentences):
        if i < len(sentences) - 2:
            pattern = " ".join(str(sentences[j]) for j in range(i, i + 3))
            result.append(pattern)

    return result


def get_dataset_from_text(embeddings: list[list[float]]) -> pd.DataFrame:
    """Converts embeddings into a pd.DataFrame with columns: `id` and `values`.

    Args:
        embeddings (list[list[float]]): Embeddings.

    Returns:
        pd.DataFrame: Dataframe with columns: `id` and `values`.
    """
    data = [
        {
            "id": f"id-{i}",
            "values": embedding,  # Ensure this is a list of floats
        }
        for i, embedding in enumerate(embeddings)
    ]
    return pd.DataFrame(data)


if __name__ == "__main__":
    prepare_dataset()
