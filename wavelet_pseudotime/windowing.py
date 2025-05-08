import numpy as np
from warnings import warn


class Window:
    def __init__(self,
                 n_windows: int):
        self.n_windows = n_windows
        self.window_centers = np.linspace(0, 1, n_windows)
        self.window_function = None
        return

    def apply(self,
              positions: np.ndarray,
              values: np.ndarray,
              exclude_minimum: bool = False):
        weights = np.zeros((self.n_windows, len(positions)))
        if exclude_minimum:
            windowed_sums = np.zeros((self.n_windows, values.shape[1]))
            for idx_col in range(values.shape[1]):
                for idx_wc, wc in enumerate(self.window_centers):
                    weights[idx_wc, :] = self.window_function(positions, wc)
                    val_min = np.min(values[:, idx_col])
                    s_vec = weights[idx_wc, :] * (values[:, idx_col] > val_min)
                    s = (s_vec).sum()
                    if s != 0:
                        weights[idx_wc, :] = s_vec / s
                windowed_sums[:, idx_col] = weights @ values[:, idx_col]
            return windowed_sums
        else:
            for idx, wc in enumerate(self.window_centers):
                weights[idx, :] = self.window_function(positions, wc)
                weights[idx, np.isnan(weights[idx,:])] = 0
                s = np.sum(weights[idx,:])
                if s > 0:
                    weights[idx, :] /= s
            return weights @ values
        return

    def get_center_window(self, num_samples=500):
        wc = self.window_centers[len(self.window_centers)//2]
        positions = np.linspace(0,1, num_samples)
        return positions, self.window_function(positions, wc)

    def get_summed_windows(self):
        positions = np.linspace(0,1,500)
        signal = np.zeros((500,))
        for wc in self.window_centers:
            signal += self.window_function(positions, wc)
        return positions, signal

    def get_each_window(self):
        positions = np.linspace(0,1,500)
        signals = []
        for wc in self.window_centers:
            signals.append(self.window_function(positions, wc))
        return positions, signals

class GaussianWindow(Window):
    def __init__(self,
                 n_windows: int,
                 sigma: float = 1.0):
        super().__init__(n_windows)
        self.sigma = sigma
        self.window_function = lambda position, center: np.exp(-((position-center) ** 2) / (2 * sigma ** 2))
        return

class RectWindow(Window):
    def __init__(self,
                 n_windows: int,
                 width: float = 0.01):
        super().__init__(n_windows)
        self.width = width
        self.window_function = lambda position, center: np.heaviside(self.width/2 - np.abs(position-center), 1)
        # Check if all parts of 0-1 are covered by the windows

        if np.min(self.get_summed_windows()) == 0:
            warn("The width and spacing of the window will exclude some samples.")
        return

class ConfinedGaussianWindow(Window):
    def __init__(self,
                 n_windows: int,
                 sigma: float = 1.0,
                 max_distance: float = 1.0):
        super().__init__(n_windows)
        self.sigma = sigma
        def window_function(position, center):
            return np.exp(-((position-center) ** 2) / (2 * sigma ** 2)) * (np.abs(position-center) <= max_distance)
        self.window_function = window_function
        return

