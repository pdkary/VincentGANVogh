from abc import ABC, abstractmethod
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow.python.keras.layers.core import Flatten

##simple builder pattern class for Dense/Convolutional model builders
class BuilderBase(ABC):
    def __init__(self,input_layer: KerasTensor):
        self.out = input_layer
        self.layer_count = 0
        self.tracked_layers = {}
    
    def build(self):
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

    @abstractmethod
    def track(self,*args,**kwargs):
        pass
