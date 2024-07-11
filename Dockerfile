#TheDockerfile is a text file that contains a list of commands that the Docker client calls while creating an image.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]