# 2.20.0-gpu was the latest, but it's borked due to CUDA/cuDNN incompatibility.
# https://github.com/tensorflow/tensorflow/issues/80538
#FROM tensorflow/tensorflow:2.20.0-gpu
FROM tensorflow/tensorflow:2.17.0-gpu

# This hides Tensorflow's verbose info logs.
ENV TF_CPP_MIN_LOG_LEVEL=3

RUN pip install swig \
  && pip install "gymnasium[classic-control,box2d]==1.2.0" Pillow

WORKDIR /app

ENTRYPOINT [ "python" ]
CMD [ "main.py" ]
