# wdflow with its compiled core, gwpy, pycbc and JupyterLab.
#
#   docker build -t wdflow .
#   docker run --rm -it -p 8888:8888 -v "$PWD:/work" wdflow
#
# The learned stage needs torch and is left out by default, since it roughly
# doubles the image:
#
#   docker build -t wdflow --build-arg WITH_TORCH=1 .
FROM python:3.12-slim

LABEL org.opencontainers.image.title="wdflow"
LABEL org.opencontainers.image.description="WDF: un-modelled transient search in the wavelet domain"
LABEL org.opencontainers.image.source="https://github.com/elenacuoco/wdflow"
LABEL org.opencontainers.image.licenses="GPL-3.0-or-later"
LABEL org.opencontainers.image.version="1.3.0"

ARG WITH_TORCH=0

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 1000 wdf

# The CPU build of torch: the default one from PyPI carries CUDA, which an
# image without a GPU never uses.
RUN if [ "$WITH_TORCH" = "1" ]; then \
        pip install "torch>=2.1" --index-url https://download.pytorch.org/whl/cpu \
        && pip install "torch_geometric>=2.5" ; \
    fi

# Only what the build reads, so the image carries no repository and no test
# fixtures.
COPY pyproject.toml README.md LICENSE /src/wdflow/
COPY wdf /src/wdflow/wdf
RUN pip install "/src/wdflow[pipeline,data,mock,tutorials]" jupyterlab \
    && rm -rf /src/wdflow

# Fail the build rather than ship an image whose core does not import:
# `wdf.analysis` works without it, so an ordinary import proves nothing.
RUN python -c "import py4tsa.tsa, wdf.analysis, wdf.processes.wdfUnitDSWorker; \
print('py4tsa and wdf import')"

USER wdf
WORKDIR /work
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.token=", "--ServerApp.root_dir=/work"]
