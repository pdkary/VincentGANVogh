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
  def __init__(self, callable: Callable,name:str, args: Dict = {}, kwargs: Dict = {}):
      super().__init__(callable, args=args, kwargs=kwargs)
      self.layer_dict = {}
      self.name = name
      self.count = 0
  
  def get(self):
    name = self.name + "_" + str(self.count)
    self.count +=1
    self.kwargs["name"] = name
    self.layer_dict[name] = self.callable(**self.args,**self.kwargs)
    return self.layer_dict[name]

class NormalizationConfig(CallableConfig):
  pass
