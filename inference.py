
import json
import logging
import os
import torch
import requests
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch.nn as nn

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(1)
    logger.info('Loading the model')
    print(2)
    model = models.resnet18(pretrained=True)
    print(3)
    num_ftrs = model.fc.in_features
    print(4)
    model.fc = nn.Linear(num_ftrs, 2)
    print(5)
    
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
#         model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(f))
    print(6)    
    model.to(device).eval()
    print(7)
    logger.info('Done loading model')
    return model


def predict_fn(input_data, model):
    logger.info('Generating prediction based on input parameters.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        return model(input_data.to(device))
