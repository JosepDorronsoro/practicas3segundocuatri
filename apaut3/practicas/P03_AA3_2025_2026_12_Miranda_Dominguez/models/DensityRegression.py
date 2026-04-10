import numpy as np
from . import parzen_density


class DensityRegression:
    
    def __init__(self, h=0.5, y_grid=None):
        self.h = h
        self.y_grid = y_grid

    def fit(self, X, y):
        """
        Fit joint density P(X,Y)
        """
       # TODO: Implementar el método fit para ajustar la densidad conjunta P(X,Y) usando el método de Parzen.
        return self

    def predict(self, X):
        """
        Predict using conditional expectation E[Y|X]
        """
        # TODO: Implementar el método predict para predecir E[Y|X] usando la densidad conjunta ajustada en el método fit.

        return np.array(preds)