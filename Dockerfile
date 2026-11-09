FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --uid 10001 --create-home cannery
COPY alembic.ini ./
COPY migrations ./migrations
USER cannery
CMD ["uvicorn", "cannery.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
