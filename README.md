# Simple Chatbot using Ollama and RAG

![alt text](assets/demo.png)

## TODO List

- Add MLX LLM
- More mode: chat, summarize...

## Setup

### Local

```bash
pip install ./src

```

### With Docker

```bash
docker compose up --build -d
```

## Run

### Local Only

```bash
python src/app.py --host localhost
```

Go to: `http://0.0.0.0:7860/` after `docker compose` completed.
