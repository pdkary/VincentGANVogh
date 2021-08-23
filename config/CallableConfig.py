from typing import Callable, Dict

class CallableConfig():
  def __init__(self,
               callable: Callable,
               args: Dict = {},
               kwargs: Dict = {}):
    self.callable = callable
    self.args = args
    self.kwargs = kwargs
  
  def get(self):
    return self.callable(**self.args,**self.kwargs)

class ActivationConfig(CallableConfig):
  pass

class NormalizationConfig(CallableConfig):
  pass
