FROM python:3.10.7-slim
MAINTAINER Damir Suleev "DaMIRka-Oo"

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

COPY . /mlops_suleev
WORKDIR /mlops_suleev

EXPOSE 5000