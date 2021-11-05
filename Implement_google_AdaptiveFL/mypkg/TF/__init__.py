from .gpu.limitGPU import setGPU
from .mymodel.cnn import CNN_Model
from .mymodel.resnet18 import myResNet
__all__ = [
    "setGPU",
    "CNN_Model",
    "myResNet",
]