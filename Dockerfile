FROM python:3.11-slim
WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY ./ /code
RUN pip install .

EXPOSE 7860

CMD ["python", "-m", "rag_chatbot", "--host", "localhost"]