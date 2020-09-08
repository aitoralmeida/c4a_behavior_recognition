FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get upgrade -y --allow-unauthenticated && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    build-essential \
    git \
    python3-dev \
    python3-pip

RUN pip3 install --upgrade pip && python3 -m pip install --upgrade pip
RUN pip3 install -U setuptools
RUN pip3 install -U keras keras-bert keras-transformer keras_pos_embd
RUN pip3 install -U tensorflow imutils
RUN pip3 install -U gensim matplotlib sklearn pandas tqdm

RUN ["mkdir", "results"]

ADD . /

VOLUME /results
