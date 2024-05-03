FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    pkg-config \
    make \
    libhdf5-dev \
    libc-dev \
    npm \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g configurable-http-proxy

RUN pip install --upgrade pip

RUN pip install h5py
RUN pip install dacapo-ml
RUN pip install notebook

RUN git clone https://github.com/janelia-cellmap/dacapo.git
RUN mv dacapo/examples examples && rm -rf dacapo

EXPOSE 8000

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8000", "--NotebookApp.allow_origin='*'", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''", "--NotebookApp.notebook_dir='/app/'"]

