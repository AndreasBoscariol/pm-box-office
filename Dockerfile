FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

CMD ["uvicorn", "pm_box_office.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
