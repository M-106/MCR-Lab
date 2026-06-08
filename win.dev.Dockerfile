FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
# FROM pytorch/pytorch:2.2.0-cpu

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

# Copy your project file for installation
COPY pyproject.toml ./
# COPY . .  
# .dockerignore limits the copied content

# Install mcrlab
RUN pip install --no-cache-dir -e .

# Default command
# CMD ["mcrlab"]
# CMD ["mcrlab", "--config", "config/config.yaml"]

# call: docker run mcrlab-img --config config/test.yaml



