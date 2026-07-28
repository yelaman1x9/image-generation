FROM python:3.11-slim-bookworm

# CPU-only wheels by default - usable, but a 512x512 image takes minutes rather than
# under a second. For an NVIDIA GPU:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 -t image-generation:gpu .
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG FAST_MODEL=stabilityai/sd-turbo
ARG BASE_MODEL=stable-diffusion-v1-5/stable-diffusion-v1-5
ARG SCORE_MODEL=openai/clip-vit-base-patch32

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    FAST_MODEL=${FAST_MODEL} \
    BASE_MODEL=${BASE_MODEL} \
    SCORE_MODEL=${SCORE_MODEL}

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the weights in so the container needs no network at runtime. fp16 only: the fp32
# copies would double the image for variants nothing here ever loads.
RUN python -c "\
import os, torch; from diffusers import AutoPipelineForText2Image;\
from transformers import CLIPModel, CLIPProcessor;\
[AutoPipelineForText2Image.from_pretrained(m, torch_dtype=torch.float16, variant='fp16')\
 for m in (os.environ['FAST_MODEL'], os.environ['BASE_MODEL'])];\
CLIPModel.from_pretrained(os.environ['SCORE_MODEL']);\
CLIPProcessor.from_pretrained(os.environ['SCORE_MODEL']);\
print('cached')"

COPY src/ src/
COPY scripts/ scripts/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "webapp:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
