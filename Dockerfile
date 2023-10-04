FROM python:3.8.12

RUN set -x && \
  apt-get update && \
  apt-get install -y --no-install-recommends libgomp1 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY . /work

RUN pip install -r requirements.txt

CMD ["bash"]