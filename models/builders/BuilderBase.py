from abc import ABC, abstractmethod
from typing import Dict
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow.keras.layers import Flatten

##simple builder pattern class for Dense/Convolutional model builders
class BuilderBase(ABC):
    def __init__(self,input_layer: KerasTensor):
        self.out = input_layer
        self.layer_count = 0
        self.view_layers = []
        self.feature_layers = []
        self.awaiting_concatenation = {}
    
    def build(self):
        if self.awaiting_concatenation != {}:
            raise Exception("ya fucked it")
        return self.out
    
    def flatten(self):
        self.out = Flatten()(self.out)
        return self

    @abstractmethod
    def block(self,*args,**kwargs):
        return self

    @abstractmethod
    def layer(self,*args,**kwargs):
        return self.out
