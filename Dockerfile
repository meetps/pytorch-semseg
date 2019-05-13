# NVIDIA Container Runtime is required
FROM nvidia/cuda:9.0-devel

RUN apt-get update && apt-get install -y \
    git python3-dev python3-tk python3-pip nano
RUN pip3 install --upgrade pip

WORKDIR /opt/app

ADD requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

RUN adduser --disabled-password --gecos "" app

RUN chown -R app:app /opt/app
USER app
