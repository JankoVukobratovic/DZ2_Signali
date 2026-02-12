from scipy.signal import butter, lfilter


def lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    coeffs = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    b = coeffs[0]
    a = coeffs[1]
    return lfilter(b, a, data)


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # 'ba' output as requested
    coeffs = butter(order, [low, high], btype='band', analog=False, output='ba')
    return lfilter(coeffs[0], coeffs[1], data)

def communication_channel(yt, fs, f_cutoff, order):
    yr = lowpass_filter(yt, f_cutoff, fs, order)
    return yr


