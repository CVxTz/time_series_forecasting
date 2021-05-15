FROM python:3.8-slim

WORKDIR /app

COPY ./time_series_forecasting ./time_series_forecasting
COPY requirements.txt requirements.txt
COPY setup.py setup.py

COPY tests tests

RUN pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
RUN pip install . --no-cache-dir
