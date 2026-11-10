FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git openssl && rm -rf /var/lib/apt/lists/*
COPY server.py /server.py
ENTRYPOINT ["python", "/server.py"]
