import asyncio
import os

from langchain_community.document_loaders import PyPDFLoader

STRINGS_TO_REPLACE = {"• ": ""}


def main():
    text = load_pdf()
    text_cleaned = replace_forbidden_chars(text)
    text_chunked = chunk_text(text_cleaned)


async def load_pdf():
    for file_name in os.listdir("data"):
        if file_name.endswith(".pdf"):
            filepath = os.path.abspath(os.path.join("data", file_name))
            loader = PyPDFLoader(filepath)
            pages = []
            async for page in loader.alazy_load():
                pages.append(page)
            print(f"{pages[0].metadata}\n")
            # print(type(pages[0].page_content))
            # print(pages[0].page_content)
            return " ".join(pages)


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


if __name__ == "__main__":
    # Run the async function
    # asyncio.run(load_pdf())
    main()
