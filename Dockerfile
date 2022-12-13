FROM python:3.10.7-slim
MAINTAINER Damir Suleev "DaMIRka-Oo"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /mlops_suleev
WORKDIR /mlops_suleev

EXPOSE 5000

CMD ["main.py"]