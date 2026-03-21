FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
# FROM pytorch/pytorch:2.2.0-cpu

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
# COPY . .

# Default command
# CMD ["python", "train.py"]




