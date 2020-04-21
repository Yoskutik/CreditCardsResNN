FROM python:3

COPY src/classify.py classify.py
COPY model.h5 model.h5
COPY data/scaler.pickle scaler.pickle
COPY requirements.txt requirements.txt

WORKDIR /

ENV TF_CPP_MIN_LOG_LEVEL 3

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "classify.py"]