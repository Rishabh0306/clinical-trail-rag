from nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Install necessary dependencies for Python and other tools
RUN apt-get update && \
    apt-get install -y \
    wget zlib1g-dev libssl-dev libffi-dev libsqlite3-dev
# Install Python 3.10 from source
RUN wget https://www.python.org/ftp/python/3.10.4/Python-3.10.4.tgz && \
    tar xvf Python-3.10.4.tgz && \
    cd Python-3.10.4 && \
    ./configure --enable-optimizations && \
    make -j "$(nproc)" && \
    make install && \
    cd .. && \
    rm -rf Python-3.10.4 Python-3.10.4.tgz

RUN mkdir clinical-trial-rag

COPY requirements.txt /clinical-trial-rag/requirements.txt
WORKDIR /clinical-trial-rag
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

COPY data/ /clinical-trial-rag/data/
COPY src/ /clinical-trial-rag/src/
COPY test_data/ /clinical-trial-rag/test_data/

EXPOSE 9090

CMD ["python3", "src/app.py"]