FROM python:3.11-slim

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

WORKDIR "/app"

COPY . .

CMD ["python3","-m","tg"]
