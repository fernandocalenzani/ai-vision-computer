FROM python:3.8

WORKDIR /app

RUN pip install opencv-python==4.8.1.78
RUN pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib


COPY . .

CMD ["python", "src/haarcascate.py"]
