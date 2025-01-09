import os
import re

import repackage
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

repackage.up()
from cfg.config import load_config
from utils.constants import ConstantsManagement

CONFIG = load_config()
CONSTANTS = ConstantsManagement()
CLIENT = OpenAI(api_key=CONFIG["openai_api_key"])


class PDFProcessor:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding(CONSTANTS.TIKTOKEN_ENCODING)
        self.text = ""
        self.embeddings = []

    def load_pdf(self) -> str:
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
        self.text = " ".join(files)

    def clean_text(self, replacement_dict: dict | None = None):
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
            replacement_dict = CONSTANTS.STRINGS_TO_REPLACE
        elif not isinstance(replacement_dict, dict):
            raise TypeError("Must be a dictionary.")
        if isinstance(self.text, str):
            # Paragraph numbers removal (eg. '08 ')
            self.text = re.sub(r"\s*\d{2}\s*", " ", self.text)
            for key in replacement_dict:
                self.text = self.text.replace(key, replacement_dict[key])

        elif isinstance(self.text, list):
            result = []
            for elem in self.text:
                if not isinstance(elem, str):
                    raise TypeError("Text must be string or list of strings.")
                # Paragraph numbers removal (eg. '08 ')
                elem = re.sub(r"\s*\d{2}\s*", " ", elem)
                for key in replacement_dict:
                    elem = elem.replace(key, replacement_dict[key])
                result.append(elem)
            self.text = result
        else:
            raise TypeError("Text must be string or list of strings.")

    def chunk_text(self):
        """Splits text and returns chunks.

        Args:
            text (str): Text to split.

        Returns:
            list[str]: Chunks after splitting.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=self.tiktoken_len,
            separators=["\n\n", "\n", " ", ""],
        )

        return text_splitter.split_text(self.text)

    def generate_embeddings(self, chunks: list[str]):
        """Generates embeddingsfor vector store.

        Args:
            chunks (list[str]): Text chunks.

        Returns:
            list[list[float]]: Embeddings.
        """
        self.embeddings = []
        for chunk in chunks:
            if isinstance(chunk, str):
                inputs = self.tokenizer.encode(chunk)
                embedding_response = CLIENT.embeddings.create(
                    input=inputs, model="text-embedding-3-small"
                )
                embedding = embedding_response.data[0].embedding
                self.embeddings.append(embedding)
            else:
                print(f"Expected string but got {type(chunk)}: {chunk}")

    def upsert_embeddings(self):
        """Upsert embeddings into Pinecone index."""
        api_key = CONFIG["pinecone_api_key"]
        index_name = CONSTANTS.PINECONE_INDEX_NAME
        environment = CONSTANTS.PINECONE_ENVIRONMENT

        # Initialize Pinecone
        pc = Pinecone(api_key=api_key, environment=environment)

        if index_name not in pc.list_indexes():
            pc.create_index(
                index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        index = pc.Index(index_name)

        # Prepare data for upsert
        data_to_upsert = [
            {"id": f"id-{i}", "values": embedding, "metadata": {"text": chunk}}
            for i, (embedding, chunk) in enumerate(
                zip(self.embeddings, self.chunk_text())
            )
        ]

        # Upsert embeddings
        index.upsert(vectors=data_to_upsert)

    def tiktoken_len(self, text) -> int:
        """Returns the length of tokens.

        Args:
            text (str): Text to apply function on.

        Returns:
            int: Length of tokens.
        """
        tokens = self.tokenizer.encode(text, disallowed_special=())
        return len(tokens)


class PDFProcessorBuilder:
    def __init__(self):
        self.processor = PDFProcessor()

    def load_pdf(self):
        """Load a PDF file into the processor."""
        self.processor.load_pdf()
        return self

    def clean_text(self):
        """Clean the text in the processor."""
        self.processor.clean_text()
        return self

    def chunk_text(self):
        """Chunk the text and return it."""
        return self.processor.chunk_text()

    def generate_embeddings(self):
        """Generate embeddings from chunks."""
        chunks = self.chunk_text()
        self.processor.generate_embeddings(chunks)
        return self

    def upsert_embeddings(self):
        """Upsert embeddings into Pinecone."""
        self.processor.upsert_embeddings()
        return self

    def build(self):
        """Return the constructed PDFProcessor object."""
        return self.processor


# Usage Example
if __name__ == "__main__":
    builder = PDFProcessorBuilder()

    # Build and process PDF
    builder.load_pdf().clean_text().generate_embeddings().upsert_embeddings()
