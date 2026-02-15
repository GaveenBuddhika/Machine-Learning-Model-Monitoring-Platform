# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install the necessary Python packages
# This will install Flask, Scikit-learn, Prometheus-client, etc.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose Port 5000 for Flask UI and 8000 for Prometheus metrics
EXPOSE 5000
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]