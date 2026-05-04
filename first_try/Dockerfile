# Python 3.10; TF pins match ai/source/ai/upstream/SETTING_SAMPLE/requirements_ubuntu.txt
FROM python:3.10-slim-bookworm
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-demo.txt app.py run.sh ./
RUN chmod +x run.sh
EXPOSE 8765
CMD ["./run.sh", "--host", "0.0.0.0", "--port", "8765"]
