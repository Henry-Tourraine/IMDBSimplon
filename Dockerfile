FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements /requirements

RUN pip install -r requirements

EXPOSE 80

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]