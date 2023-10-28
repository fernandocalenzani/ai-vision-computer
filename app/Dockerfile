FROM python:3.8

WORKDIR /app

RUN pip install opencv-python==4.8.1.78

COPY . .

CMD ["python", "src/haarcascate.py"]
