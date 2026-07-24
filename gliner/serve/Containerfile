FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    ray[serve] \
    transformers \
    huggingface_hub \
    safetensors \
    flair

COPY . /app/gliner
RUN pip install --no-cache-dir /app/gliner

COPY gliner/serve/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV TOKENIZERS_PARALLELISM=true

EXPOSE 8000

# Default configuration (override with -e)
ENV GLINER_MODEL="urchade/gliner_small-v2.1"
ENV GLINER_DEVICE="cuda"
ENV GLINER_DTYPE="bfloat16"
ENV GLINER_MAX_BATCH_SIZE="32"

ENTRYPOINT ["/entrypoint.sh"]
