FROM python:3.10-slim

# Set up non-root user (Hugging Face Spaces default UID 1000)
RUN useradd -m -u 1000 user

WORKDIR /app

# Install system build dependencies required by C-extensions (LightGBM, Scikit-learn, SHAP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY --chown=user:user . .

# Set permissions for local SQLite database and model caching
RUN chown -R user:user /app

# Switch to non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Launch FastAPI web server on port 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
