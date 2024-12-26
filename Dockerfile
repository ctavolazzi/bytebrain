# Use Python 3.10 as base image
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (updated installation)
RUN curl https://ollama.ai/install.sh | sh

# Set working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock ./
COPY nova_system/ ./nova_system/
COPY bots/ ./bots/
COPY auto_chat/ ./auto_chat/
COPY examples/ ./examples/
COPY ollama_client.py ./
COPY README.md ./

# Create directories for history and logs
RUN mkdir -p history logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=http://localhost:11434

# Development stage
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN poetry install --no-interaction --no-ansi

# Production stage
FROM base as production

# Install production dependencies only
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

# Create entrypoint script
RUN echo '#!/bin/bash\n\
# Start Ollama in the background\n\
ollama serve &\n\
\n\
# Wait for Ollama to start\n\
echo "Waiting for Ollama to start..."\n\
until curl -s http://localhost:11434/api/tags > /dev/null; do\n\
    sleep 1\n\
done\n\
\n\
# Pull the default model\n\
echo "Pulling llama3.2 model..."\n\
ollama pull llama3.2\n\
\n\
# Start the chat application\n\
exec python examples/nova_chat.py "$@"' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]