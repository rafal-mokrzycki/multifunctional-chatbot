class Constants:
    PINECONE_ENVIRONMENT = "us-east-1"
    PINECONE_INDEX_NAME = "british-airways"
    TIKTOKEN_ENCODING = "cl100k_base"
    OPENAI_MODEL_NAME = "o1-mini"
    STRINGS_TO_REPLACE = {"• ": "", "\n": " ", "  ": " ", "   ": " "}
    SPECIFIC_KEYWORDS = ["British Airways", "Flight", "Flight Ticket"]


# ConstantsManagement class
class ConstantsManagement:
    def __init__(self):
        # Set constants from separate classes as attributes
        for cls in [Constants]:
            for key, value in cls.__dict__.items():
                if not key.startswith("__"):
                    self.__dict__.update(**{key: value})

    def __setattr__(self, name, value):
        raise TypeError("Constants are immutable")
