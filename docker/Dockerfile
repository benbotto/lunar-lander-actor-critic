# 2.20.0-gpu was the latest, but it's borked due to CUDA/cuDNN incompatibility.
# https://github.com/tensorflow/tensorflow/issues/80538
#FROM tensorflow/tensorflow:2.20.0-gpu
FROM tensorflow/tensorflow:2.17.0-gpu

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# This hides Tensorflow's verbose info logs.
ENV TF_CPP_MIN_LOG_LEVEL=3

# Dependencies for glvnd and X11.
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libxext6 \
    libx11-6 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    freeglut3-dev \
  && rm -rf /var/lib/apt/lists/*

RUN pip install swig \
  && pip install "gymnasium[classic-control,box2d]==1.2.0"

WORKDIR /app

ENTRYPOINT [ "python" ]
CMD [ "main.py" ]
