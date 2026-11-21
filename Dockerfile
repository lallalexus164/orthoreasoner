FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .
CMD ["python", "-m", "orthoreasoner_rc.cli.train", "--config", "configs/experiments/smoke.yaml"]
