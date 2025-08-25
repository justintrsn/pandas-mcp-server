# Multi-stage build with UV for faster builds
FROM python:3.10-slim as builder

# Install build dependencies and curl for UV installation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV and create venv in one RUN command to preserve PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:$PATH" && \
    /root/.local/bin/uv venv /opt/venv

# Set environment for subsequent commands
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:/root/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install with UV
COPY requirements.txt .
RUN /root/.local/bin/uv pip install -r requirements.txt && \
    /root/.local/bin/uv pip install mcp openai python-dotenv

# Final stage
FROM python:3.10-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# MCP Server Configuration - 0.0.0.0 is CORRECT for Huawei ModelArts
ENV MCP_SERVER_HOST=0.0.0.0
ENV MCP_SERVER_PORT=8000
ENV MCP_SERVER_TRANSPORT=sse
ENV MCP_LOG_LEVEL=INFO

# Create non-root user for security
RUN useradd -m -u 1000 mcp && \
    mkdir -p /app/logs /app/data /app/charts && \
    chown -R mcp:mcp /app

WORKDIR /app

# Copy application code
COPY --chown=mcp:mcp . .

# Create necessary directories
RUN mkdir -p logs data charts test && \
    chown -R mcp:mcp /app

# Switch to non-root user
USER mcp

# Expose the MCP server port
EXPOSE 8000

# Default command runs the server
CMD ["python", "server.py"]