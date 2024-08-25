FROM python:3.10

WORKDIR /code

COPY requirements.txt /code
RUN pip install -r requirements.txt

COPY model/ /code/model/
COPY app/ /code/model/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8088" ]
