# Simple Chatbot using Ollama and RAG

![alt text](assets/demo.png)

## Setup

### Install Ollama

Download at: https://ollama.com/

### Use OpenAI model

Create file `.env` and input:

```bash
OPENAI_API_KEY = "[YOUR API KEY]"
```

### Local

```bash
pip install .

```

### With Docker

```bash
docker compose up
```

## Run

### Step 1: (Local Only)

```bash
python app.py --host localhost
```

### Step 2:

Go to: `http://0.0.0.0:7860/`.
