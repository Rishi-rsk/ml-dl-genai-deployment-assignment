FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install flask numpy tensorflow==2.10 gunicorn

EXPOSE 8001

CMD ["gunicorn", "--bind", "0.0.0.0:8001", "app:app"]
