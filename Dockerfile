# syntax=docker/dockerfile:1.0
FROM ubuntu
WORKDIR /code

RUN apt-get update && apt-get install -y python3.11 python3-pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["python","./src/task_1"]