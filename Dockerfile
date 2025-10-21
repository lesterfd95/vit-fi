# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# install python and pip 
RUN apt-get update && apt-get install -y python3 python3-pip

# project dependencies 
COPY requirements.txt /
RUN pip install -r requirements.txt

# project folder
WORKDIR /app
