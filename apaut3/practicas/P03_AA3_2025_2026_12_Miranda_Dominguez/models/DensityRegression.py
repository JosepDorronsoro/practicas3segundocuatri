import numpy as np
from .parzen_density import parzen_density

class DensityRegression:
    
    def __init__(self, h=0.5, y_grid=None):
        self.h = h
        self.y_grid = y_grid

    def fit(self, X, y):
        y_col = y.reshape(-1, 1) if y.ndim == 1 else y
        XY = np.concatenate([X, y_col], axis=1)
        self.joint_density_ = parzen_density(XY, self.h)

        if self.y_grid is None:
            self.y_grid_ = np.linspace(y.min() - 1, y.max() + 1, 100)
        else:
            self.y_grid_ = self.y_grid

        return self
    
    def predict_map(self, X):
        """argmax_y P(X,Y) — máximo a posteriori"""
        preds = []
        for x in X:
            p_xy = self._eval_grid(x)
            preds.append(self.y_grid_[np.argmax(p_xy)])
        return np.array(preds)

    def predict_mean(self, X):
        """E[Y|X] — esperanza condicional"""
        preds = []
        for x in X:
            p_xy = self._eval_grid(x)
            p_x  = np.sum(p_xy)
            if p_x < 1e-12:
                preds.append(0.0)
            else:
                preds.append(np.sum(self.y_grid_ * p_xy) / p_x)
        return np.array(preds)

    def predict_median(self, X):
        """Mediana condicional — valor de y donde la CDF condicional cruza 0.5"""
        preds = []
        for x in X:
            p_xy = self._eval_grid(x)
            p_x  = np.sum(p_xy)
            if p_x < 1e-12:
                preds.append(0.0)
            else:
                cdf = np.cumsum(p_xy) / p_x
                idx = np.searchsorted(cdf, 0.5)
                idx = min(idx, len(self.y_grid_) - 1)
                preds.append(self.y_grid_[idx])
        return np.array(preds)

    def _eval_grid(self, x):
        """Evalúa P(X=x, Y=y) usando la densidad de Parzen para todos los valores de y_grid."""
        x_tiled = np.tile(x, (len(self.y_grid_), 1))
        y_col   = self.y_grid_.reshape(-1, 1)
        xy_grid = np.concatenate([x_tiled, y_col], axis=1)
        return self.joint_density_(xy_grid)