FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "crypto_predictor/dashboard/dashboard_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
