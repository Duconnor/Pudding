from abc import ABC, abstractmethod

class _BaseModel(ABC):
    '''
    This is the base class for machine learning models implemented in Pudding.
    '''

    @abstractmethod
    def fit(self, X, y=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X, **kwargs):
        raise NotImplementedError
