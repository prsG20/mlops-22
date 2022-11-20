# FROM ubuntu:latest
FROM python:3.10.4
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
COPY ./api/app.py /exp/api/app.py
COPY ./svm_gamma=0.001_C=0.7.joblib /exp/svm_gamma=0.001_C=0.7.joblib
RUN pip3 install -U scikit-learn
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
#COPY ./models /exp/models
#COPY ./results /exp/results
WORKDIR /exp
EXPOSE 5000
ENV CLF_NAME = "svm"
ENV RANDOM_STATE = 42
VOLUME ./models
CMD python plot_graph.py --clf_name ${CLF_NAME} --random_state ${RANDOM_STATE}
#CMD ["python3", "plot_graph.py", "--clf_name", ${CLF_NAME}, "--random_state" ${RANDOM_STATE}]
#CMD ["python3","./api/app.py"]

# for build: docker build -t exp:v1 -f docker/Dockerfile .
# for running a container: docker run -p 5000:5000 -it exp:v1
# for running a container: docker run -p 5000:5000 -it exp:v1 --volume ./models ./results

#docker build -f docker/Dockerfile -t final:v1 --build-arg CLF_NAME="svm" --build-arg RANDOM_STATE=42 .

#docker run -it -e CLF_NAME="svm" -e RANDOM_STATE="32" final:v1