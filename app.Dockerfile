from python:3.10
RUN mkdir clinical-trial-rag

COPY requirements.txt /clinical-trial-rag/requirements.txt
WORKDIR /clinical-trial-rag
RUN pip install -r requirements.txt

COPY data/ /clinical-trial-rag/data/
COPY src/ /clinical-trial-rag/src/
COPY test_data/ /clinical-trial-rag/test_data/
COPY models/ /clinical-trial-rag/models/

EXPOSE 9090

CMD bash
