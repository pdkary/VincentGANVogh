from enum import Enum

class SearchableEnum(Enum):
    @classmethod
    def get_index_by_name(cls,name:str):
        for i,n in enumerate(cls.__dict__['_member_names_']):
            if n == name:
                return i
        return -1

    @classmethod
    def get_index_by_value(cls,val):
        for i,m in enumerate(cls.__dict__['_value2member_map_']):
            if m == val:
                return i
        return -1
    
    @classmethod
    def get_by_index(cls,i:int):
        if i >= len(cls.__dict__['_member_names_']):
            raise IndexError("index out of range")
        return list(cls.__dict__['_member_map_'].values())[i]
  
    @classmethod
    def get_value_by_index(cls,i:int):
        return cls.get_by_index(i).value
  
    @classmethod
    def get_name_by_index(cls,i:int):
        return cls.get_by_index(i).name
