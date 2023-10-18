import torch
import numpy as np
import triton_python_backend_utils as pb_utils  #Why is this not a pip module?
from transformers import FlavaModel, FlavaFeatureExtractor, BertTokenizer

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import OrderedDict
import torch




class TritonPythonModel:
    

    def initialize(self, args):
        self.flava = FlavaModel.from_pretrained("facebook/flava-full")
        self.fe = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")
        self.tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")
        self.flava = self.flava.cuda().eval()

    def execute(self, requests):
        responses = []

        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "image")
            # tokenizer or list of strings
            texts = pb_utils.get_input_tensor_by_name(request, "text") #I want this to be a list of strings and then show similarity score between each string and the given one image

            
            image_input = self.fe(inp, return_tensors="pt").to("cuda")
            text_tokens = self.tokenizer(texts, return_tensors="pt", padding=True, max_length=77).to("cuda")

            with torch.no_grad():
                # We take the output embedding for the CLS token for both encoders
                image_features = self.flava.get_image_features(**image_input)[:, 0].float()
                text_features = self.flava.get_text_features(**text_tokens)[:, 0].float()

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
            # extract the number from matrix; logging eqn of print
            

            inference_response = pb_utils.InferenceResponse(similarity) #should this of only tensor form??
            responses.append(inference_response)
        return responses
    
    