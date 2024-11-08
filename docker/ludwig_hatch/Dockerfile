FROM python:3.12

ENV PATH="/root/.local/bin:$PATH"
RUN apt-get -y update
RUN apt-get -y install pipx
RUN apt-get -y install git libsndfile1 build-essential g++ cmake ffmpeg sox libsox-dev
RUN pipx ensurepath --force
RUN pipx install hatch
RUN python3 -m pip install --upgrade pipx
WORKDIR /ludwig
#COPY /ludwig/ .
COPY . .

RUN hatch env create
RUN hatch build

ENTRYPOINT ["ludwig"]