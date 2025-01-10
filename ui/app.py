from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.main import handle_conversation
from utils.logger import CustomLogger

app = FastAPI(reload=True)


# Define a Pydantic model for the request body
class QuestionRequest(BaseModel):
    question: str  # Expecting a field named 'question'


# Store conversation history in memory (for simplicity)
context = ""
log_file_path = None  # Initialize log file path globally

# Serve static files (CSS) from the static directory
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

# Set up Jinja2 templates for rendering HTML files
templates = Jinja2Templates(directory="ui/templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serve the main HTML page for the chatbot application.

    This endpoint handles GET requests to the root URL ("/") and
    returns the main HTML template for the chatbot interface. It uses
    Jinja2 templates to render the HTML file.

    Parameters:
    ----------
    request : Request
        The incoming request object containing information about the
        HTTP request.

    Returns:
    -------
    HTMLResponse
        The rendered HTML response containing the chatbot interface.

    Example:
    --------
    When a user navigates to the root URL, this function will return
    the index.html template, allowing them to interact with the
    chatbot.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
async def ask(request: QuestionRequest):
    """
    Handle user questions sent to the chatbot.

    This endpoint processes POST requests to the "/ask" URL. It
    expects a JSON payload containing a user's question. The function
    invokes the handle_conversation function to generate a response
    based on the current context and the provided question. The
    conversation history is updated with each new question and answer.

    Parameters:
    ----------
    request : QuestionRequest
        A Pydantic model instance that contains the user's question as
        a string.

    Returns:
    -------
    dict
        A JSON object containing the chatbot's response under the key
        "answer".

    Example:
    --------
    When a user sends a POST request with a question:

        {
            "question": "What is your name?"
        }

    The function will return a response like:

        {
            "answer": "I am your AI chatbot!"
        }
    """
    global context, log_file_path
    custom_logger = CustomLogger(log_file_path)

    # Create a new log file for each session if it doesn't exist
    custom_logger.create_log_file()
    custom_logger.create_directory()

    answer = handle_conversation(context, request.question)

    # Log the conversation to the file
    custom_logger.write_logs(request.question, answer)

    context += f"\nUser: {request.question}\nAI: {answer}"

    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
