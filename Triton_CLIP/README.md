
# NVIDIA Triton Flava Test Setup

*Note: I updated the Triton version to 23.04 as this image version was already available on dpl02 (and so I could omit the download). There is no specific need for this version, so feel free to change it back to any version you already have on your machine.*

### First container:
Make sure you are in the directory `Triton/` where the `Dockerfile` is located, too. Then, run
```
docker build -t tritonserver-clip . 
```
to build a Docker image for the Triton server. 

*Note: As we need some Python packages in the Backend, we can't use the standard Triton Inference Server Container.*

After a successful build process, you can start the server with
```
docker run --gpus=1 --rm -p8003:8000 -p8004:8001 -p8005:8002 --shm-size=512M -v ${PWD}/models:/models tritonserver-clip
```

### Second container:
The second container is not strictly necessary, but I tried to stick the previous setup, so you can start it in a separate console via
```
docker run -it --add-host host.docker.internal:host-gateway -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:22.01-py3-sdk bash
```
*Note: Inside the second container, you cannot access the endpoint of the Triton server via `localhost:8003`. (The reason for that is that `localhost` inside the container is different from `localhost` on the host machine.) A quick fix is to add the `--add-host host.docker.internal:host-gateway` flag. Inside the container, you can then access the host's "localhost" via `host.docker.internal`. A more elegant solution can be achieved by using Docker compose to orchestrate the two containers. (Details omitted)*

Inside the second container, simply run `python client.py`. It should currently say something like
```
image.shape after adding batch dim: (1, 640, 480, 3)
text.shape after adding batch dim: (1, 1)
The similarity score of this image with text is  0.27882057428359985
```
in the console output.

### A tip for debugging: Model (un)loading

You can unload the model from a running Triton server via 
```
curl -v -X POST http://localhost:8003/v2/repository/models/model/unload
```
(and you can load it again by replacing "unload" with "load").