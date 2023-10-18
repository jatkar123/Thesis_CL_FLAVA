import argparse

import numpy as np
import requests
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import *


def main():
    client = httpclient.InferenceServerClient(url="localhost:8003") #tritonserver:http

    # Inputs
    url = "http://images.cocodataset.org/val2017/000000161642.jpg"
    text = "clock"
    image = np.asarray(Image.open(requests.get(url, stream=True).raw)).astype(
        np.float32
    )
    image = np.expand_dims(image, axis=0)

    # Set Inputs
    input_tensors = [httpclient.InferInput("image", image.shape, datatype="FP32")]
    #input_tensors = [httpclient.InferInput("text", text.shape, datatype="STRING")] how to define this text type??
    input_tensors[0].set_data_from_numpy(image)

    # Set outputs
    outputs = [httpclient.InferRequestedOutput("similarity")]

    # Query
    query_response = client.infer(
        model_name="trial", inputs=input_tensors, outputs=outputs
    )

    # Output
    similarity_score = query_response.as_numpy("similarity")
    print("the similarity score of this image with text is ", similarity_score)


if __name__ == "__main__":
    main()