FROM nvcr.io/nvidia/pytorch:22.03-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 tmux git -y

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt