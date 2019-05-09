# NVIDIA Container Runtime is required
FROM nvidia/cuda:9.0-devel

RUN apt-get update && apt-get install -y \
    git python3-dev python3-tk python3-pip

WORKDIR /opt/semseg

RUN pip3 install --upgrade pip
ADD requirements.txt /opt/semseg/requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "train.py", "--config", "configs/fcn8s_pascal.yml"]
