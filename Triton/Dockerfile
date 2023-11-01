FROM nvcr.io/nvidia/tritonserver:23.04-py3

RUN pip install torch
RUN pip install pillow 

CMD bash -c "tritonserver --model-repository=/models --model-control-mode=explicit --load-model model --min-supported-compute-capability=5.2"