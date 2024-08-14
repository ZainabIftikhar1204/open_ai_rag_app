FROM python:3.10-slim

# Install build tools
RUN apt-get update && apt-get install -y build-essential

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn main:app --reload --port=8000 --host=0.0.0.0
