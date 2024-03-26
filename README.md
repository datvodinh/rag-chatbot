# Simple Chatbot using Ollama and RAG

![alt text](assets/demo.png)

## Setup

### Install Ollama

Download at: https://ollama.com/

### Local

```bash
pip install ./src

```

### With Docker

```bash
docker compose up --build -d
```

## Run

### Step 1: (Local Only)

```bash
python src/app.py --host localhost
```

### Step 2:

Go to: `http://0.0.0.0:7860/`.
