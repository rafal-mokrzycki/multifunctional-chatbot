# Multifunctional Chatbot
Chatbot to get information from text files using LLMs.

## Installation

1. Run:

```
python -m venv .venv
source .venv/bin/activate # (.venv\Scripts\activate for Windows)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. For Linux:

```
curl https://ollama.ai/install.sh | sh
ollama pull mistral
ollama run llama3.2:1b
```

3. For Windows:
- Go to [https://ollama.com/download](https://ollama.com/download) and download Ollama for your OS.
- Run the installation script.
- For Windows make sure you added a system environmental variable `OLLAMA_MODELS`: `{folder}`.
- Run `ollama pull mistral` in the command line inside virtual environment.
- Run `ollama run llama3.2:1b` in order to install a language model.

## Usage

1. Activate virtual environment: `.venv\Scripts\activate`.
2. Run app: `python -m  ui.app` and go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.
3. Alternatively, run app like so: `python models/ollama/main.py` and talk to the bot via terminal.
