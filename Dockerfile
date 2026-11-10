FROM python:3.12-slim AS app
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --uid 10001 --create-home cannery
RUN mkdir -p /home/cannery/artifacts && chown cannery:cannery /home/cannery/artifacts
COPY alembic.ini ./
COPY migrations ./migrations
USER cannery
CMD ["uvicorn", "cannery.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

FROM app AS registrar
COPY --from=docker:29-cli /usr/local/bin/docker /usr/bin/docker
USER root
