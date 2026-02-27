FROM python:3.12-slim

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
        libsndfile1 \
        ffmpeg \
        sox \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

USER $USERNAME
ENV PATH="/home/$USERNAME/.local/bin:$PATH"

# Pre-install pip tools so editable install is fast
RUN pip install --user --no-cache-dir --upgrade pip setuptools wheel
