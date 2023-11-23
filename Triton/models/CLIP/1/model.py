import torch
import numpy as np
import triton_python_backend_utils as pb_utils  # Why is this not a pip module?
#from transformers import FlavaModel, FlavaFeatureExtractor, BertTokenizer
from transformers import AutoProcessor, CLIPModel, AutoTokenizer, CLIPProcessor

# import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import OrderedDict
import torch


class TritonPythonModel:
    def initialize(self, args):
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
        #self.fe = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.fe = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        #self.clip = self.clip.cuda().eval()

        self.logger = pb_utils.Logger

    def execute(self, requests):
        responses = []

        for request in requests:
            image = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()
            

            text = pb_utils.get_input_tensor_by_name(request, "text").as_numpy()
            self.logger.log_info("Shape of text: " + str(text.shape))
            self.logger.log_info(str(text))
            texts = [t.decode("UTF-8") for t in text.tolist()[0]]

            self.logger.log_info(str(texts))

            image_input = self.fe(text = texts, images = image, return_tensors="pt", padding=True).to("cuda")

            outputs = self.clip(**image_input)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score

            similarity = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            
        
            self.logger.log_info(
                "Similarity before returning InferenceResponse:\n" + str(similarity.T)
            )

            inference_response = pb_utils.InferenceResponse(
                [pb_utils.Tensor("similarity", similarity.cpu().detach().numpy())]
            )
            responses.append(inference_response)
            
        return responses