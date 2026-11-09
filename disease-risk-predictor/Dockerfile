FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY risk ./risk
COPY data ./data
COPY app ./app

# Train the model while building the image, so the container is ready to serve.
RUN python -m risk.train --no-plots

RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
