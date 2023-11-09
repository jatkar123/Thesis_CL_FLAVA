
# NVIDIA Triton Flava Test Setup

*Note: I updated the Triton version to 23.04 as this image version was already available on dpl02 (and so I could omit the download). There is no specific need for this version, so feel free to change it back to any version you already have on your machine.*

### First container:
Make sure you are in the directory `Triton/` where the `Dockerfile` is located, too. Then, run
```
docker build -t tritonserver-flava . 
```
to build a Docker image for the Triton server. 

*Note: As we need some Python packages in the Backend, we can't use the standard Triton Inference Server Container.*

After a successful build process, you can start the server with
```
docker run --gpus=1 --rm -p8003:8000 -p8004:8001 -p8005:8002 --shm-size=512M -v ${PWD}/models:/models tritonserver-flava
```

### Second container:
The second container is not strictly necessary, but I tried to stick the previous setup, so you can start it in a separate console via
```
docker run -it --add-host host.docker.internal:host-gateway -v ${PWD}/Triton:/workspace/ nvcr.io/nvidia/tritonserver:23.04-py3-sdk bash
```
Then, install some packages for the client:
```
pip install -r requirements.txt
```

*Note: Inside the second container, you cannot access the endpoint of the Triton server via `localhost:8003`. (The reason for that is that `localhost` inside the container is different from `localhost` on the host machine.) A quick fix is to add the `--add-host host.docker.internal:host-gateway` flag. Inside the container, you can then access the host's "localhost" via `host.docker.internal`. A more elegant solution can be achieved by using Docker compose to orchestrate the two containers. (Details omitted)*

Inside the second container, simply run `python client.py`. It should give you the predicted classes of the three images 

![Airplane 1](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.Of5n8Mk257GxvWQAhgn5pAHaEM%26pid%3DApi&f=1&ipt=83a6d7af258db93aa13f9850ebe8ca711bb61b99729afb4ea546e340b720f59c&ipo=images)

![Airplane 2](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.c6bwfJFG5wWYxUj4tl1N7gHaE2%26pid%3DApi&f=1&ipt=23140604025fba006eab0078a36dddf291af6cc42dc0800fc186f2dba27ef419&ipo=images)

![Airplane 3](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.IA-C43FHXiXH5bKAlbmBhQHaEo%26pid%3DApi&f=1&ipt=741c858eff026dc10ebeb670c7403f66cb56ccc55a343517c449d0a10d056a0a&ipo=images)

in the console.

### A tip for debugging: Model (un)loading

You can unload the model from a running Triton server via 
```
curl -v -X POST http://localhost:8003/v2/repository/models/model/unload
```
(and you can load it again by replacing "unload" with "load").