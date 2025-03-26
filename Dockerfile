ARG BASE_IMAGE=nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMAGE} as dev-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /src

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND noninteractive\
    SHELL=/bin/bash
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends\
    wget\
    bash\
    openssh-server &&\
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

RUN apt-get update && apt-get install -y --no-install-recommends
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install git -y
RUN apt-get install python3.10 -y
RUN apt-get install python3-pip -y
RUN apt-get install python3.10-distutils -y

RUN apt-get install python3.10-dev -y
RUN apt-get install python3.10-venv -y
RUN python3.10 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install system dependencies
RUN apt-get install python3-opencv -y

# Install Python dependencies
RUN python3.10 -m pip install torch torchvision torchaudio
RUN python3.10 -m pip install runpod


RUN git clone https://github.com/Zheng-Chong/CatVTON.git
WORKDIR CatVTON
RUN python3.10 -m pip install diffusers==0.32.2
RUN python3.10 -m pip install fvcore==0.1.5.post20221221
RUN python3.10 -m pip install av==14.2.0
RUN python3.10 -m pip install opencv-python==4.11.0.86
RUN python3.10 -m pip install omegaconf==2.3.0
RUN python3.10 -m pip install pycocotools==2.0.8
RUN python3.10 -m pip install scipy==1.15.2
RUN python3.10 -m pip install accelerate==1.5.2
RUN python3.10 -m pip install transformers==4.49.0
RUN python3.10 -m pip install cloudpickle==3.1.1
RUN python3.10 -m pip install omegaconf==2.3.0
# Download weights
COPY builder/download_weights.py .
RUN python3.10 download_weights.py
RUN rm download_weights.py

COPY src/runpod_infer.py .
COPY src/prediction.py .


CMD python3.10 -u runpod_infer.py
