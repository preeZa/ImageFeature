FROM python:3.9
WORKDIR /ImageFeature
COPY ./requirements.txt /ImageFeature/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /ImageFeature/requirements.txt

COPY ./app /ImageFeature/app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

ENV PYTHONPATH="${PYTHONPATH}:/ImageFeature"

CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]