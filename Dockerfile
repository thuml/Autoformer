FROM nvidia/cuda:11.2.0-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update  -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    unrar \
    unzip \
    git && \
    apt-get upgrade -y libstdc++6 && \
    apt-get clean -y

RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y gcc-9 && \
    apt-get upgrade -y libstdc++6

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b  && \
    rm -rf Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN conda install -n base -c conda-forge mamba

ADD ./environment.yml ./environment.yml
RUN mamba env update -n base -f ./environment.yml && \
    conda clean -afy

