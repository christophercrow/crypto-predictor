# Dockerfile

FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire project
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Set Streamlit to run the dashboard by default
CMD ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
