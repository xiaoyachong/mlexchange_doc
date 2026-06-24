FROM python:3.11-slim

RUN useradd -m appuser

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libhdf5-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install deps + both tiled_agentic and als_knowledge_agent (declared in pyproject.toml)
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY als_knowledge_agent/src/ ./als_knowledge_agent/src/
RUN pip install --no-cache-dir --prefer-binary -e .

# Copy rest of the project
COPY . .

# Ensure als_knowledge_agent is always importable regardless of editable install wiring
ENV PYTHONPATH=/app/als_knowledge_agent/src

RUN chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "uvicorn", "tiled_agentic.api_server:app", \
     "--host", "0.0.0.0", "--port", "8002"]