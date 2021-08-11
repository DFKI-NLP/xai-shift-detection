FROM tensorflow/tensorflow:2.0.4-gpu
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install dvipng texlive-latex-recommended texlive-fonts-recommended texlive-fonts-extra wget python3-pip hdf5-tools libcairo2-dev pkg-config python3-dev libgirepository1.0-dev vim nano xorg openbox --assume-yes git
RUN pip3 install alibi-detect
ENV CUDA_HOME /usr/local/cuda

ADD . /home/failing-loudly/

WORKDIR /home/failing-loudly/
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/josipd/torch-two-sample
WORKDIR /home/failing-loudly/torch-two-sample/
RUN python3 setup.py install
WORKDIR /home/failing-loudly/
RUN rm -r saved_models