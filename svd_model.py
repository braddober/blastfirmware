__author__ = 'gjones'

import numpy as np
from matplotlib import pyplot as plt


class SVDNoiseModel(object):
    def __init__(self, data, sampling_rate=256e6 / 2 ** 14):
        self.data = data
        self.sampling_rate = sampling_rate

    def construct_model(self, n_components=1, nfft=2 ** 15):
        data_fft = np.fft.rfft(self.data, axis=0)
        U, S, V = np.linalg.svd(data_fft[:nfft, :], full_matrices=False)
        common_mode_fft = np.dot(np.dot(U[:, :n_components], np.diag(S[:n_components])), np.conj(V[:n_components, :]))
        full_common_mode_fft = np.empty_like(data_fft)
        full_common_mode_fft[:nfft, :] = common_mode_fft
        self.common_mode = np.fft.irfft(full_common_mode_fft, axis=0)
        self.corrected_data = self.data - self.common_mode