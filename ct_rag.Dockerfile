from python:3.10
RUN mkdir /home/test/clinical-trial-rag

COPY requirements.txt /home/test/clinical-trial-rag/requirements.txt
WORKDIR /home/test/clinical-trial-rag
RUN pip install -r requirements.txt

COPY data/ /home/test/clinical-trial-rag/data/
COPY src/ /home/test/clinical-trial-rag/src/
COPY test_data/ /home/test/clinical-trial-rag/test_data/

EXPOSE 9090

CMD bash
