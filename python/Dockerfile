FROM docker.io/bitnami/python:3.10

WORKDIR /root/notebooks

ADD requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 1234
