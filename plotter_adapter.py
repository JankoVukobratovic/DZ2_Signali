import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import freqz
def close_all():
    plt.close('all')

def plot_time_and_freq(t, signal, fs, title="Signal Analysis"):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=3)

    ax1.plot(t, signal, color='blue')
    ax1.set_title(f"{title} - Vremenski Domen")
    ax1.set_xlabel("Vreme [s]")
    ax1.set_ylabel("Amplituda")
    ax1.grid(True)

    n = len(signal)
    freqs = np.fft.fftfreq(n, 1 / fs)[:n // 2]
    magnitude = np.abs(np.fft.fft(signal))[:n // 2] * (2 / n)

    ax2.plot(freqs, magnitude, color='red')
    ax2.set_title(f"{title} - Frekfencijski Domen")
    ax2.set_xlabel("Frekfencija [Hz]")
    ax2.set_ylabel("Magnituda ")
    ax2.grid(True)

    plt.show()


def plot_filter_response(b, a, fs, title="Filter Frequency Response"):
    w, h = freqz(b, a, worN=8000)
    freqs = w * fs / (2 * np.pi)

    fig, ax1 = plt.subplots()

    ax1.plot(freqs, 20 * np.log10(np.abs(h)), 'b')
    ax1.set_title(title)
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.grid(True)

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h)) * (180 / np.pi)
    ax2.plot(freqs, angles, 'g--')
    ax2.set_ylabel('Phase [degrees]', color='g')

    plt.show()