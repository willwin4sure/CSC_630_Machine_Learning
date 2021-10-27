import torch

resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)