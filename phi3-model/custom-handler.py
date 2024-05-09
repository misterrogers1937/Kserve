# handler.py

import torch
import torchvision
import logging
import transformers
import os
import json

from ts.torch_handler.base_handler import BaseHandler
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger('MyTorch')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('torch.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)
logger.info("Transformers version %s", transformers.__version__)

class ModelHandler(BaseHandler):

    def initialize(self, context):
        """Initialize function loads the model and the tokenizer

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model or
            tokenizer is missing

        """

        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")

        # use GPU if available
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        logger.info(f'Using device {self.device}')

        # load the model
        model_file = self.manifest['model']['modelFile']
        model_path = os.path.join(model_dir, model_file)

        if os.path.isfile(model_path):
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f'Successfully loaded model from {model_file}')
        else:
            raise RuntimeError('Missing the model file')

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer is not None:
            logger.info('Successfully loaded tokenizer')
        else:
            raise RuntimeError('Missing tokenizer')

        self.pipe = pipeline(
             "text-generation",
             model=self.model,
             tokenizer=self.tokenizer,
        )

        self.initialized = True

    def preprocess(self, requests):
        """Tokenize the input text using the suitable tokenizer and convert
        it to tensor

        Args:
            requests: A list containing a dictionary, might be in the form
            of [{'body': json_file}] or [{'data': json_file}]
        """

        logger.info('Received request');
        logger.info(requests);
        # unpack the data
        data = requests[0].get('body')
        if data is None:
            data = requests[0].get('data')


        logger.info(f'Data value : {data}')
        return data

    def inference(self, messages):
        """Predict class using the model

        Args:
            inputs: tensor of tokenized data
        """

        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        logger.info(f'Requested prediction for {messages}')
        print(type(messages))
        messages_list =[messages]
        print(type(messages_list))
        output = self.pipe(messages_list, **generation_args)
        logger.info(f'Predictions 1 successfully created. {output}')

        #messages = [{"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}]
        #output = self.pipe(messages, **generation_args)
        #logger.info(f'Predictions 2 successfully created. {output}')

        return output

    def postprocess(self, output):
        """
        Convert the output to the string label provided in the label mapper (index_to_name.json)

        Args:
            outputs (list): The integer label produced by the model

        Returns:
            List: The post process function returns a list of the predicted output.
        """

        return output
