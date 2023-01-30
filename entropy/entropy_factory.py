from abc import ABC,abstractmethod

class EntropyHandler(ABC):
    @abstractmethod
    def insert_pattern(self,p):
        pass
    @abstractmethod
    def get_entropy(self,l,last_key=0):
        pass