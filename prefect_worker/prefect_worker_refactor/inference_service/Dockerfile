# Inference service image: the only image in this repo that contains PyTorch.
# Pick the torch/CUDA base that matches the models you serve.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.29" "mlflow==2.22.0" numpy
COPY inference_service/ /app/inference_service/

# Non-root user
RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8091
# One worker per GPU: each worker holds its own model cache
CMD ["uvicorn", "inference_service.main:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "1"]
