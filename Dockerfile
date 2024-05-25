FROM python:3.11-slim
WORKDIR /code
COPY ./ /code
RUN pip install .

EXPOSE 7860

CMD ["python", "-m", "rag_chatbot", "--host", "host.docker.internal"]