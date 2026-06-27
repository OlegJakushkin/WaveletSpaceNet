# WaveletSpaceNet — GPU image.  Requires the NVIDIA Container Toolkit on the host.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace
ENV PYTHONUNBUFFERED=1
