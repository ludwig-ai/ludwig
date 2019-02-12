FROM tensorflow/tensorflow:latest-py3

RUN apt-get install -y --no-install-recommends git
RUN git clone https://github.com/uber/ludwig.git
RUN cd ludwig/ && pip install -r requirements.txt && python -m spacy download en && python setup.py install
