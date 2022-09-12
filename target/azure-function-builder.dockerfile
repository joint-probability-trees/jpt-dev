FROM mcr.microsoft.com/azure-functions/python:4-python3.8
RUN apt update \
    && apt install -y make g++ libatlas-base-dev gfortran libxslt1-dev libxml2-dev \
    && ln -s /usr/local/bin/python3.8 /usr/bin/python3.8 \
    && pip3.8 install virtualenv
WORKDIR /app
ENTRYPOINT make wheel