FROM python:3.9-slim

WORKDIR /app

ENV BASE_DIR=/app
ENV PYTHONPATH=${BASE_DIR}/src
ENV PORT=80

COPY ./requirements.txt ${BASE_DIR}/requirements.txt
COPY ./src ${BASE_DIR}/src

RUN pip install --no-cache-dir --upgrade -r ${BASE_DIR}/requirements.txt
RUN pip install --no-cache-dir --upgrade -r ${BASE_DIR}/src/requirements.txt
