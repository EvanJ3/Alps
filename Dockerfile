# Dockerfile, Image, Container
FROM continuumio/miniconda3

WORKDIR /Usopc-Alpine-App

ADD environment.yml .

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "UsopcAlpine", "/bin/bash", "-c"]

ADD ./App ./App

ENTRYPOINT ["conda", "run", "-n", "UsopcAlpine", "python", "App/app.py", "-docker"]


