# wdflow, with p4TSA's compiled core built in.
#
# The core is built from source rather than installed from an index: p4TSA has
# no PyPI distribution, because the frame-file library it reads GWF with has no
# wheel there, and the unrelated `pytsa` package on PyPI is a decorator library
# that would resolve in its place. So the image starts from conda-forge, which
# does carry framel, and compiles the core against it.
#
#   docker build -t wdflow .
#   docker run --rm -it -p 8888:8888 -v "$PWD:/work" wdflow
#
# The learned stage needs torch and is left out by default, since it roughly
# doubles the image:
#
#   docker build -t wdflow --build-arg WITH_TORCH=1 .
FROM mambaorg/micromamba:1.5.8

LABEL org.opencontainers.image.title="wdflow"
LABEL org.opencontainers.image.description="WDF: un-modelled transient search in the wavelet domain"
LABEL org.opencontainers.image.source="https://github.com/elenacuoco/wdflow"
LABEL org.opencontainers.image.licenses="GPL-3.0-or-later"
LABEL org.opencontainers.image.version="1.1.1"

# Which p4TSA to build. A tag or a commit, not a branch: an image that builds a
# different core depending on the day is not reproducible.
ARG P4TSA_REF=v2.2.0
ARG WITH_TORCH=0

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER docker/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

# Every RUN below needs the environment on the path.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN if [ "$WITH_TORCH" = "1" ]; then \
        micromamba install -y -n base -c conda-forge \
            "pytorch>=2.1" pytorch-cpu "torch-geometric>=2.5" \
        && micromamba clean --all --yes ; \
    fi

# The compiled core. Its own build isolation is off so that it builds against
# the conda environment's gsl, fftw and framel rather than fetching its own.
RUN git clone --depth 1 --branch "$P4TSA_REF" \
        https://github.com/elenacuoco/p4TSA.git /tmp/p4TSA \
    && pip install --no-build-isolation -v /tmp/p4TSA \
    && rm -rf /tmp/p4TSA

COPY --chown=$MAMBA_USER:$MAMBA_USER . /src/wdflow
RUN pip install --no-deps /src/wdflow

# Fail the build rather than ship an image whose core does not import: a broken
# pytsa is invisible until the first run otherwise, and `wdf.analysis` works
# without it, so an ordinary import proves nothing.
RUN python -c "import pytsa.tsa, wdf.analysis, wdf.processes.wdfUnitDSWorker; \
print('pytsa and wdf import')"

WORKDIR /work
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.token=", "--ServerApp.root_dir=/work"]
