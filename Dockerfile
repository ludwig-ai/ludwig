FROM tensorflow/tensorflow:latest-py3

RUN git clone --depth=1 https://github.com/uber/ludwig.git \
    && cd ludwig/ \
    && pip install -r requirements.txt -r requirements_text.txt \
          -r requirements_image.txt -r requirements_audio.txt \
          -r requirements_serve.txt -r requirements_viz.txt
    && python setup.py install

WORKDIR /data

ENTRYPOINT ["ludwig"]
