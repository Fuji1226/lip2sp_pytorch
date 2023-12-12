FROM python:3.8.12

RUN set -x && \
  apt-get update && \
  apt-get install -y cmake ffmpeg flac && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /root/lip2sp_pytorch
COPY ./requirements.txt /root/lip2sp_pytorch

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["bash"]