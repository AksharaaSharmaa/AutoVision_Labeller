# -------------------------
# Base image with Python
# -------------------------
FROM python:3.11-slim

# -------------------------
# Set working directory
# -------------------------
WORKDIR /app

# -------------------------
# Install system dependencies for OpenCV + fonts + git
# -------------------------
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Copy requirements.txt
# -------------------------
COPY requirements.txt .

# -------------------------
# Install Python dependencies
# -------------------------
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Copy app files
# -------------------------
COPY . .

# -------------------------
# Expose Streamlit port
# -------------------------
EXPOSE 8501

# -------------------------
# Set Streamlit config to avoid auto-opening browser
# -------------------------
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_PORT=8501

# -------------------------
# Start Streamlit
# -------------------------
CMD ["streamlit", "run", "app.py"]
