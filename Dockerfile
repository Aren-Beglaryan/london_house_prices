FROM python:3.8.12-slim-buster

WORKDIR /app

COPY ./src /app/src

COPY ./service /app/service

RUN pip install --upgrade pip \
    && pip install -r /app/src/requirements.txt \
    && pip install -r /app/service/requirements.txt

WORKDIR /app/service

CMD ["python", "app.py"]
