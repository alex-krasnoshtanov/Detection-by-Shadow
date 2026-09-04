# Two stages, so the final image carries a built virtualenv and none of the
# build tooling.
#
# The torch install is separated from everything else for two reasons. It is by
# far the largest layer, so it wants to be cached independently of the source.
# And it has to come from PyTorch's CPU index: the default PyPI wheel bundles
# CUDA and would add roughly 2 GB to an image that will almost never see a GPU.
# Installing it first means the project install below finds torch already
# satisfied and does not reach for the CUDA build.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

# The dependency list lives in pyproject.toml and only there. Duplicating it
# here would drift the moment either changed.
#
# LICENSE and README.md are not optional extras: pyproject points `license` and
# `readme` at them, so hatchling fails metadata generation outright if either is
# missing from the build context.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN /opt/venv/bin/pip install ".[demo]"


FROM python:3.12-slim

LABEL org.opencontainers.image.title="Detection by Shadow" \
      org.opencontainers.image.description="Locate an off-frame pedestrian from the shadow they cast into the frame." \
      org.opencontainers.image.source="https://github.com/alex-krasnoshtanov/Detection-by-Shadow" \
      org.opencontainers.image.licenses="MIT"

COPY --from=builder /opt/venv /opt/venv

# SHADOW_MODEL_DIR points at a mountable volume so a restart does not re-fetch
# ~95 MB from the release.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SHADOW_MODEL_DIR=/models \
    OMP_NUM_THREADS=4

RUN useradd --create-home --uid 10001 demo \
    && mkdir -p /models \
    && chown demo:demo /models
USER demo

VOLUME ["/models"]
EXPOSE 8000

# The model is fetched on first start, so a fresh container is briefly unready
# rather than broken; start-period covers that download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "shadow_detection.demo.app:app", "--host", "0.0.0.0", "--port", "8000"]
