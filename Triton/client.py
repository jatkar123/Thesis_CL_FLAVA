import argparse

import numpy as np
import requests
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import *


def main():
    client = httpclient.InferenceServerClient(url="host.docker.internal:8003")

    # Inputs
    url = "http://images.cocodataset.org/val2017/000000161642.jpg"

    image = np.asarray(Image.open(requests.get(url, stream=True).raw)).astype(
        np.float32
    )
    image = np.expand_dims(image, axis=0)
    print("image.shape after adding batch dim:", image.shape)

    text = "clock"
    text = np.array([text], dtype="object").reshape((1,))
    text = np.expand_dims(text, axis=0)
    print("text.shape after adding batch dim:", text.shape)

    # Set Inputs
    input_tensors = [
        httpclient.InferInput("image", image.shape, datatype="FP32"),
        httpclient.InferInput("text", text.shape, np_to_triton_dtype(text.dtype)),
    ]
    input_tensors[0].set_data_from_numpy(image, binary_data=True)
    input_tensors[1].set_data_from_numpy(text, binary_data=True)

    # Set outputs
    outputs = [httpclient.InferRequestedOutput("similarity")]

    # Query
    query_response = client.infer(
        model_name="model", inputs=input_tensors, outputs=outputs
    )

    # Output
    similarity_score = query_response.as_numpy("similarity").item()
    print("The similarity score of this image with text is ", similarity_score)


if __name__ == "__main__":
    main()
