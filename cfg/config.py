import json
import os
from pathlib import Path


def load_config(file_name="config.json") -> dict:
    """
    Loads config.json file with configuration for the app; config.json
    has to be in the same directory as config.py

    Args:
        file_name (str, optional): Config file name.
        Defaults to "config.json".

    Returns:
        dict: Dictionary with configuration.
    """
    file_path = Path(__file__).parent.joinpath(file_name)
    if not os.path.exists(file_path):
        create_config()
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_config(file_name="config_.json"):
    file_path = Path(__file__).parent.joinpath(file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        openai_api_key = input("Enter OpenAI API key: ")
        openai_organization = input("Enter OpenAI organization name: ")
        openai_project = input("Enter OpenAI project name: ")
        pinecone_api_key = input("Enter Pinecone API key: ")
        data = {
            "openai_api_key": openai_api_key,
            "openai_organization": openai_organization,
            "openai_project": openai_project,
            "pinecone_api_key": pinecone_api_key,
        }
        json.dump(data, f, ensure_ascii=False, indent=4)
