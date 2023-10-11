
FROM python:3.7-slim-buster

RUN apt-get update && apt-get install -y git

#RUN pip3 install git+https://github.com/evidentlyai/evidently.git



ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

#COPY metrics_app .
WORKDIR /opt/program
COPY requirements.txt .

RUN pip3 install -r requirements.txt

#RUN pip3 install evidently==0.3.1 

ENTRYPOINT ["python3"]
