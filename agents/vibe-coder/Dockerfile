FROM python:3.11-slim
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git --no-install-recommends && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "main"]
