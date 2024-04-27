# 🤖 Chat with multiple PDFs, using Ollama and LlamaIndex

![alt text](assets/demo.png)

# 📖 Table of Contents

- [`Feature`](#⭐️-features)
- [`Setup`](#💻-setup)
  - [`Kaggle`](#1-kaggle-recommended)
  - [`Local`](#2-local)
    - [`Clone`](#21-clone-project)
    - [`Docker`](#22-docker)
    - [`On machine`](#23-on-machine)
- [`Todo`](#🎯-todo)

# ⭐️ Features

- Run locally or Kaggle (new)
- Using any model from `Huggingface` and `Ollama`
- Chat with multiples PDFs.
- Chat with multiples languages (Coming soon).
- Simple UI with `Gradio`.

# 💻 Setup

## 1. Kaggle (Recommended)

- Import [`notebooks/kaggle.ipynb`](notebooks/kaggle.ipynb) to Kaggle
- Replace `<YOUR_NGROK_TOKEN>` with your tokens.

## 2. Local

### 2.1. Clone project

```bash
git clone https://github.com/datvodinh/rag-chatbot.git
cd rag-chatbot
```

### 2.2. Docker

```bash
docker compose up --build
```

### 2.3. On machine

#### 2.3.1 Install Ollama

- Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

- MacOS

- [Download](https://ollama.com/)

#### 2.3.3 Install Package

```bash
source ./scripts/install.sh
```

#### 2.3.4 Run

```bash
source ./scripts/run.sh
```

or

```bash
python app.py --host localhost
```

- Using Ngrok

```bash
source ./scripts/run.sh --ngrok
```

### 3. Go to: `http://0.0.0.0:7860/` or Ngrok link after setup completed!

## 🎯 Todo

- Support better Embedding Model for Vietnamese and other languages.
- Knowledge Graph (for Structure Data).
- Better Document Processing.
- MLX model.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=datvodinh/rag-chatbot&type=Date)](https://star-history.com/#datvodinh/rag-chatbot&Date)