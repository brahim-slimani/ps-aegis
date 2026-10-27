from python:3.12.15

WORKDIR /aegis

COPY app/ ./app
COPY data/ ./data
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt 
RUN python -m app.scripts.ingest --recreate

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
