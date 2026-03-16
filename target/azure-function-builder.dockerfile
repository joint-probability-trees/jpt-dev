FROM mcr.microsoft.com/azure-functions/python:4.8.0.b-python3.10
RUN apt install -y make g++
RUN ln -s /usr/local/bin/python3.10 /usr/bin/python3.10
RUN pip3.10 install -U pip virtualenv
WORKDIR /app
ENTRYPOINT make wheel