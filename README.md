# Chat with multiple PDFs, using Ollama and LlamaIndex

![alt text](assets/demo.png)

## Table of Contents

<details>

<summary>Click to show</summary>
  
- [`Setup`](#setup)
  - [`Install Ollama`](#install-ollama)
  - [`Local`](#local)
  - [`Docker`](#docker)
- [`Run`](#run)
- [`Todo`](#todo)
- [`Star History`](#star-history)

</details>

## Features

- Fully local model from `Huggingface` and `Ollama`
- Chat with multiples PDFs.
- Chat with multiples languages (Coming soon).
- Simple UI with `Gradio`.

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

### Docker

#### Build

```bash
docker compose up --build
```

## Run

```bash
source ./scripts/run.sh
```

or

```bash
python app.py --host localhost
```

### Using Ngrok

```bash
source ./scripts/run.sh --ngrok
```

#### Go to: `http://0.0.0.0:7860/` or Ngrok link after setup completed!

## Todo

- Support better Embedding Model for Vietnamese and other languages.
- Knowledge Graph (for Structure Data).
- Better Document Processing.
- MLX model.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=datvodinh/rag-chatbot&type=Date)](https://star-history.com/#datvodinh/rag-chatbot&Date)