import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fftfreq
from scipy.fftpack import fft

from scipy import signal

def fourier_transform(x):
    return fft(x)


def get_spectrum(signal, fs):
    n = len(signal)
    mag = np.abs(fft(signal)) / n
    freqs = fftfreq(n, 1 / fs)
    # Return only the positive half (the physical frequencies)
    half = n // 2
    return freqs[:half], mag[:half]

