FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL files including static directory
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 clara
RUN chown -R clara:clara /app
USER clara

# Expose port (Hugging Face uses 7860)
EXPOSE 7860

# Run ClaraGPT
CMD ["python", "main.py"]
