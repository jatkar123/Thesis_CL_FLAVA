import argparse

import numpy as np
import requests
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import *

from scipy.special import softmax
import plotext as plt


def classify_image_by_url(url):
    client = httpclient.InferenceServerClient(url="host.docker.internal:8003")

    image = np.asarray(Image.open(requests.get(url, stream=True).raw)).astype(
        np.float32
    )
    image = np.resize(image, (640, 480, 3)).astype(np.float32)
    image = np.expand_dims(image, axis=0)

    print("image.shape after adding batch dim:", image.shape)

    text_list = ["Cat", "Airplane", "Dog", "Car", "Sky", "Bird", "Cloud"]
    text = np.array(text_list, dtype="object").reshape((len(text_list),))
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
    similarity_score = query_response.as_numpy("similarity")[0]
    print("Similarity score shape:", similarity_score.shape)
    print(
        "The similarity score of this image with",
        text_list,
        "is",
        similarity_score,
        sep="\n",
    )

    text_probs = softmax(similarity_score * 100.0, axis=-1)
    print("Class probabilities:", text_probs)

    plt.simple_bar(text_list, text_probs, width=100)
    plt.title("Class probabilities")
    plt.show()


if __name__ == "__main__":
    for index, url in enumerate(
        {
            "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.Of5n8Mk257GxvWQAhgn5pAHaEM%26pid%3DApi&f=1&ipt=83a6d7af258db93aa13f9850ebe8ca711bb61b99729afb4ea546e340b720f59c&ipo=images",
            "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.c6bwfJFG5wWYxUj4tl1N7gHaE2%26pid%3DApi&f=1&ipt=23140604025fba006eab0078a36dddf291af6cc42dc0800fc186f2dba27ef419&ipo=images",
            "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.IA-C43FHXiXH5bKAlbmBhQHaEo%26pid%3DApi&f=1&ipt=741c858eff026dc10ebeb670c7403f66cb56ccc55a343517c449d0a10d056a0a&ipo=images",
        }
    ):
        print(f"\n\n===================== IMAGE {index+1} =====================")
        classify_image_by_url(url)
