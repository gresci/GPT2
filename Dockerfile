FROM tensorflow/tensorflow:1.13.1-gpu-py3

RUN pip3 install regex ai-integration==1.0.6

WORKDIR /model

RUN mkdir PrettyBig
ADD https://deepai-opensource-codebases-models.s3-us-west-2.amazonaws.com/gpt2-prettybig/PrettyBig/checkpoint PrettyBig/checkpoint
ADD https://deepai-opensource-codebases-models.s3-us-west-2.amazonaws.com/gpt2-prettybig/PrettyBig/model.ckpt.index PrettyBig/model.ckpt.index
ADD https://deepai-opensource-codebases-models.s3-us-west-2.amazonaws.com/gpt2-prettybig/PrettyBig/model.ckpt.meta PrettyBig/model.ckpt.meta
ADD https://deepai-opensource-codebases-models.s3-us-west-2.amazonaws.com/gpt2-prettybig/PrettyBig/model.ckpt.data-00000-of-00001 PrettyBig/model.ckpt.data-00000-of-00001

COPY . /model


