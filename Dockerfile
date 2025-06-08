FROM python:3.10-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get upgrade -y && apt-get clean && \
	pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.allow-websocket-origin=OmkarDhekane-wheatCropClassifier.hf.space"]
