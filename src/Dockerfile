FROM python:3.11-slim
WORKDIR /code
RUN export HNSWLIB_NO_NATIVE=1 
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY ./ /code
RUN pip install .

EXPOSE 7860

CMD ["gradio","src/app.py"]