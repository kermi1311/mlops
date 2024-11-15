# Use the full Python 3.8 image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies required for TensorFlow
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libsndfile1 \
    libatlas-base-dev \
    gfortran \
    libblas-dev \
    liblapack-dev \
    && apt-get clean

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
