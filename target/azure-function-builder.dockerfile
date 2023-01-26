FROM mcr.microsoft.com/azure-functions/python:4.8.0.b-python3.8
RUN apt update \
    && apt install -y make g++ libatlas-base-dev gfortran libxslt1-dev libxml2-dev
RUN ln -s /usr/local/bin/python3.8 /usr/bin/python3.8
RUN pip3.8 install -U pip virtualenv
WORKDIR /app
ENTRYPOINT make wheel