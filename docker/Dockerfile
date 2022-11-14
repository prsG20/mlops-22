# FROM ubuntu:latest
FROM python:3.10.4
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
COPY ./api/app.py /exp/api/app.py
COPY ./svm_gamma=0.001_C=0.7.joblib /exp/svm_gamma=0.001_C=0.7.joblib
RUN pip3 install -U scikit-learn
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
EXPOSE 5000
# CMD ["python3", "./plot_graph.py"]
CMD ["python3","./api/app.py"]
# for build: docker build -t exp:v1 -f docker/Dockerfile .
# for running a container: docker run -p 5000:5000 -it exp:v1