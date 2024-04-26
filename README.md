# Chat with multiple PDFs, using Ollama and LlamaIndex

![alt text](assets/demo.png)

## Setup

### Install Ollama

#### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### MacOS

- [Download](https://ollama.com/)

### Clone project

```bash
git clone https://github.com/datvodinh/rag-chatbot.git
cd rag-chatbot
```

### Local

#### Install

```bash
source ./scripts/install.sh
```

#### Run

```bash
source ./scripts/run.sh
```

or

```bash
python app.py --host localhost
```

#### Using Ngrok

```bash
source ./scripts/run.sh --ngrok
```

#### Go to: `http://0.0.0.0:7860/` or Ngrok link after setup completed!

### Docker

#### Build

```bash
docker compose up --build
```

#### Go to: `http://0.0.0.0:7860/` after setup completed!

## Todo List

- Support better Embedding Model for Vietnamese.
- Knowledge Graph (for Structure Data).
- Better Document Processing.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=datvodinh/rag-chatbot.git&type=Timeline)](https://star-history.com/#datvodinh/rag-chatbot.git&Timeline)