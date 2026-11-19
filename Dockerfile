FROM python:3.12-slim

# grpcio and pymilvus may need a compiler on platforms without a prebuilt wheel.
# Removed in the same layer so it does not stay in the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && pip install --no-cache-dir --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential

COPY . .

# Run as a non-root user. The image carries API keys at runtime, so a process that does
# not need to be root should not be.
RUN useradd --create-home --uid 10001 skillgpt \
    && chown -R skillgpt:skillgpt /app
USER skillgpt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fs http://localhost:8000/api/health || exit 1

# `python -m app.main` imports the module, builds the FastAPI object and exits: there is
# no server in it. The previous Dockerfile did exactly that, and the container only ever
# appeared to work because docker-compose overrides the command. An image has to run on
# its own, which is what the build job in CI now checks.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
