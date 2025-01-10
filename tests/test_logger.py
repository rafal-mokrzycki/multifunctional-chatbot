import os
import shutil

import pytest

from utils.logger import CustomLogger


@pytest.fixture(scope="module")
def log_dir():
    """Fixture to create log_dir for testing."""
    yield "logs_for_tests"


@pytest.fixture(scope="module")
def log_file_path(log_dir):
    """Fixture to create log_file_path for testing."""
    yield f"{log_dir}/conversation_test.log"


@pytest.fixture(scope="module")
def logger(log_file_path):
    """Fixture to create a CustomLogger instance for testing."""
    logger_instance = CustomLogger(log_file_path)

    yield logger_instance


def test_create_directory(logger, log_dir):
    """
    Test that the 'logs_for_tests' directory
    is created if it doesn't already exist.
    """

    logger.create_directory(log_dir)
    assert os.path.exists(log_dir)
    shutil.rmtree(log_dir, ignore_errors=True)


def test_create_log_file(logger, log_dir, log_file_path):
    """Test that a log file is created with a specified name."""

    # Ensure the log directory exists before creating the log file
    logger.create_directory(log_dir)

    # Pass the directory to create the log file
    logger.create_log_file(log_dir)
    assert logger.log_file_path == log_file_path
    shutil.rmtree(log_dir, ignore_errors=True)


def test_write_logs(logger, log_dir):
    """Test that logs are written correctly to the log file."""

    # Ensure the log directory exists before creating the log file
    logger.create_directory(log_dir)

    # Pass the directory to create the log file
    logger.create_log_file(log_dir)

    # Write a log entry
    logger.write_logs("What is your name?", "I am an AI chatbot.")

    # Read the log file content to verify it was written correctly
    with open(logger.log_file_path, "r", encoding="utf-8") as log_file:
        content = log_file.read()

    expected_content = "User: What is your name?\nAI: I am an AI chatbot.\n"

    assert (
        expected_content in content
    ), f"Expected content not found in log file. Content: {content}"
    shutil.rmtree(log_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main()
