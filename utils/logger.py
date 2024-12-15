import datetime
import os


class CustomLogger:
    """
    A custom logger for recording user questions and AI responses in a log file.

    Attributes:
    ----------
    log_file_path : str
        The path to the log file where conversation logs will be stored.
    """

    def __init__(self, log_file_path: str):
        """
        Initializes the CustomLogger with a specified log file path.

        Parameters:
        ----------
        log_file_path : str
            The path to the log file where logs will be written.
        """
        self.log_file_path = log_file_path

    def create_directory(self):
        """
        Creates a directory named 'logs' if it does not already exist.

        This method checks for the existence of the 'logs' directory
        and creates it if it is not found. This is necessary to store
        log files in a dedicated location.
        """
        if not os.path.exists("logs"):
            os.makedirs("logs")

    def create_log_file(self):
        """
        Creates a new log file with a timestamped name if no log file path is set.

        This method generates a timestamped filename for the log file
        based on the current date and time, and assigns it to
        `self.log_file_path`. If `self.log_file_path` is already set,
        this method does nothing.
        """
        if self.log_file_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file_path = os.path.join("logs", f"conversation_{timestamp}.log")

    def write_logs(self, question: str, answer: str):
        """
        Appends a user question and AI response to the log file.

        Parameters:
        ----------
        question : str
            The question posed by the user.

        answer : str
            The response generated by the AI.

        This method opens the specified log file in append mode and writes
        the user question and AI response to it. Each entry is formatted
        with "User: " followed by the question, and "AI: " followed by
        the response.
        """
        with open(self.log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"User: {question}\n")
            log_file.write(f"AI: {answer}\n")