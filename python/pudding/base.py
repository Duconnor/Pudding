from abc import ABC, abstractmethod

class _BaseModel(ABC):
    '''
    This is the base class for machine learning models implemented in Pudding.
    For unsupervised learning algorithms, only the fit() method will be implemented. For supervised learning algorithms, both of these two methods will be implemented.
    '''

    @abstractmethod
    def fit(self, X, y=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X, **kwargs):
        raise NotImplementedError
