FROM ludwigai/ludwig-ray:master

USER root
# Copy library scripts to execute
RUN curl -o /tmp/common-debian.sh https://raw.githubusercontent.com/microsoft/vscode-dev-containers/main/script-library/common-debian.sh

# [Option] Install zsh
ARG INSTALL_ZSH="false"
# [Option] Upgrade OS packages to their latest versions
ARG UPGRADE_PACKAGES="false"
# Install needed packages and use existing non-root user (ray).
ARG USERNAME=automatic
ARG USER_UID=1000
ARG USER_GID=102
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    # Remove imagemagick due to https://security-tracker.debian.org/tracker/CVE-2019-10131
    && apt-get purge -y imagemagick imagemagick-6-common \
    # Install common packages, non-root user
    && bash /tmp/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "true" \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip install pre-commit black pylint flake8 mypy

USER ${USER_UID}
