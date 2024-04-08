import numpy as np
import numpy.fft as fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def filter_data(data, threshold):
    """
    Apply a frequency filter to the input data.

    Args:
    - data (numpy.ndarray): Input 1D data array.
    - threshold (float): Threshold value for filtering.

    Returns:
    - numpy.ndarray: Filtered data array after applying inverse FFT.
    - numpy.ndarray: Filtered frequency components.
    """

    freq = fft.rfft(data)
    freq[np.abs(freq) < threshold] = 0
    
    return fft.irfft(freq), freq

class DataProcessor:
    def __init__(self):
        self.pca_model = None
        self.sc = None
        self.n_components = None
        self.modes_w = 0
        self.modes_b = 0
        self.modes_freqs = None
        self.freq_scale = 1
    
    def fit(self, x, y, n_components, threshold, normalize):
        """
        Apply all the procedures of data analysis, inluding
        1. apply PCA to the dataset

        Args:
        - x (numpy.ndarray): Time steps with shape (B, ).
        - y (numpy.ndarray): The input data with shape (B, C).
        - n_components (int): Number of components to keep.
        - normalize (bool): if True, each channel of data will be normalized with zero mean and unit variance.
        - threshold (bool): values below which will be clipped in frequency space.

        Returns:
        - numpy.ndarray: Transformed data after PCA.
        """

        y_pca = self.apply_pca(x=x, 
                     y=y,
                     n_components=n_components, 
                     normalize=normalize)
        # print(y_pca.shape)

        modes_filtered = np.zeros_like(y_pca)
        mode_freqs = [None] * n_components
        for i in range(y_pca.shape[1]):
            modes_filtered[:, i], freq = filter_data(y_pca[:, i], threshold)
            mode_freqs[i] = freq

        self.modes_freqs = np.stack(mode_freqs, axis=-1) # (T/2, N)
        self.freq_scale = (x[-1] - x[0]) / (x.shape[0]+1)

        return modes_filtered

    def pred_modes(self, x):
        '''
        Using the previously learned frequency representation of data, 
        predict the modes given the input variable x.

        Args:
        - x (numpy.ndarray): 1D time step

        Returns:
        - numpy.ndarray: Predicted principal components (modes) obtained from PCA with shape (B, n_components).
        '''
        modes_pred = np.zeros([x.shape[0], self.n_components])
        for n_mode in range(self.n_components):
            # get the frequency representation of the nth mode
            freq_data = self.modes_freqs[:, n_mode]

            # extract nonzero frequency data
            nonzero_idx = np.nonzero(freq_data)[0] 
            nonzero_vals = freq_data[nonzero_idx]
            freq_num = freq_data.shape[0]
            
            for idx, val in zip(np.array(nonzero_idx), nonzero_vals):
                amp = np.abs(val) / freq_num
                ang = np.angle(val)
                freq = 2*np.pi*idx/(2*freq_num) / self.freq_scale
            
                
                modes_pred[:, n_mode] += amp * np.cos(freq * x + ang)

        return modes_pred
        

    def inversePCA(self, x, y_modes):
        """
        Recover the original data from the principal components (modes) after applying PCA.

        Args:
        - x (numpy.ndarray): Time steps
        - y_modes (numpy.ndarray): Principal components (modes) obtained from PCA wit hshape (B, n_components).

        Returns:
        - numpy.ndarray: Recovered original data.
        """
        self._check_pca_model_fitted()
        assert (y_modes.shape[1] == self.n_components)

        # add the linear components 
        y_modes = y_modes + self.w * x.reshape(-1, 1) + self.b

        recovered_data = self.pca_model.inverse_transform(y_modes)

        if self.sc:
            recovered_data = self.sc.inverse_transform(recovered_data)

        return recovered_data

    def apply_pca(self, x, y, n_components, scaling=False, normalize=True):
            """
            Apply Principal Component Analysis (PCA) to the data.

            Args:
            - x (numpy.ndarray): Time steps with shape (B, ).
            - y (numpy.ndarray): The input data with shape (B, C).
            - n_components (int): Number of components to keep.
            - scaling (bool): if True, each channel of data will be normalized with zero mean and unit variance.
            - normalize (bool): if True, the method considers only the stationary modes, where the linear growth rate is zero.
            Returns:
            - numpy.ndarray: Transformed data after PCA.
            """
            if scaling:
                self.sc = StandardScaler()
                data_normalized = self.sc.fit_transform(y)
            else:
                data_normalized = y

            # perform PCA analysis
            self.n_components = n_components
            self.pca_model = PCA(n_components=n_components)
            y_pca = self.pca_model.fit_transform(data_normalized)
            self.w = np.zeros([1, n_components])
            self.b = np.zeros([1, n_components])
            if normalize:
                # apply linear regression to each mode
                for i in range(n_components):
                    self.w[0, i], self.b[0, i] = np.polyfit(x, y_pca[:, i], 1)
                
                # normalize the data according to the fitted linear representation
                y_pca = y_pca - self.w * x.reshape(-1, 1) - self.b

            return y_pca
    
    def predict(self, x):
        '''
        Predict outputs given the input time steps

        Args:
        - x (numpy.ndarray): 1D time step array

        Returns:
        - numpy.ndarray: Predicted data...
        '''
        modes_pred = self.pred_modes(x)
        pred = self.inversePCA(x, modes_pred)
        return pred


    def _check_pca_model_fitted(self):
        if self.pca_model is None:
            raise ValueError("PCA model has not been fitted yet.")

