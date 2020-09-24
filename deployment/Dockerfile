FROM ubuntu:18.04

RUN apt-get update && \
    apt-get -y install curl python3-pip && \
    apt-get -y remove python-apt-common && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install -r requirements.txt && \
    rm -r ~/.cache/pip

ENV PYTHONPATH "${PYTHONPATH}:/ma_project"