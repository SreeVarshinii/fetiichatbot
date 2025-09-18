FROM python:3.11-slim

# system deps for psycopg
RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# healthcheck (optional)
HEALTHCHECK CMD curl -f http://localhost:8501/_stcore/health || exit 1
