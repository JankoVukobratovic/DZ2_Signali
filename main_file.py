import numpy as np
import scipy
from matplotlib import pyplot as plt
import filters
import plotter_adapter
import plotter_adapter as pa
from functools import lru_cache
import utils
# P = 0

# region Zadatak 1
@lru_cache(maxsize=None)
def zadatak_1_a():
    t, fs, signal = utils.load_signal('Prvi zadatak/flauta_C5 (523Hz).wav')

    start_idx = int(len(signal) // 2)
    end_idx = start_idx + int(0.02 * fs)

    pa.plot_time_and_freq(t, signal, fs, title="Ceo Signal")

    plt.figure()
    plt.plot(t[start_idx:end_idx], signal[start_idx:end_idx])
    plt.title("Uvecana slika")
    plt.xlabel("Vreme [s]")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.show()

@lru_cache(maxsize=None)
def zadatak_1_b(play_audio = False):
    t_orig, fs, signal_orig = utils.load_signal('Prvi zadatak/flauta_C5 (523Hz).wav')

    freq_note = 523
    duration = 2.0
    t_sine, signal_sine = utils.generate_sine_alternative(freq_note, duration, fs)
    start_idx = int(len(signal_sine) // 2)
    end_idx = start_idx + int(0.02 * fs)
    pa.plot_time_and_freq(t_sine, signal_sine, fs, title="Čist sinusni signal (523Hz)")

    plt.figure()
    plt.plot(t_sine[start_idx:end_idx], signal_sine[start_idx:end_idx])
    plt.title("Uvecana slika")
    plt.xlabel("Vreme [s]")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.show()

    if play_audio:
    #print("Playing original flute...")
        utils.play_audio(signal_orig, fs)

    #print("Playing sine wave...")
        utils.play_audio(signal_sine, fs)


def zadatak_1_v():
    fs = 8000
    duration = 2.0
    cache_file = "vowel_cache_v.wav"

    vowel_signal = utils.record_with_cache(cache_file, duration, fs)

    t_full = np.linspace(0, len(vowel_signal) / fs, len(vowel_signal), endpoint=False)

    mid = len(vowel_signal) // 2
    offset = int(0.025 * fs)  # 25ms

    vowel_window = vowel_signal[mid - offset: mid + offset]
    t_window = t_full[mid - offset: mid + offset]

    pa.plot_time_and_freq(t_window, vowel_window, fs, title="Samoglasnik A - 50ms")

    #pronalazenje osnovne ucestalosti
    n = len(vowel_signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    magnituda = np.abs(np.fft.rfft(vowel_signal))

    peaks: np.ndarray
    peaks, _  = scipy.signal.find_peaks(magnituda, height=np.max(magnituda) * 0.3)
    if peaks.shape[0] > 0:
        print(f"Osnovna učestalost nađena pomoću frekfencijskog domena: {freqs[peaks[0]]}")
    else:
        print(f"Nije uspelo nalaženje osnovne učestalosti pomoću frekfencijskog domena")


# endregion

SHOULD_SAVE_FILES = False
SAVE_DIR = "steps/"

SHOW_ALL_STEPS = False
F_CUTOFF = 7000
F_CARRIER = 15000
F_SAMPLE = 44100 # at least expected
NYQUIST = F_SAMPLE / 2
MAGIC_NUMBER = 1
BP_LOW = F_CARRIER - F_CUTOFF
BP_HIGH = F_CARRIER + F_CUTOFF



FILTER_ORDER = 6

DEBUG = True
def load_2_signals_no_recording(play_signals_on_load = False, show_spectre_of_loaded_signals = True):
    t, fs, y1 = utils.load_signal('Prvi zadatak/klavir_G4 (392Hz).wav')
    _, _, y2 = utils.load_signal('Prvi zadatak/flauta_C5 (523Hz).wav')

    min_len = min(len(y1), len(y2))
    y1, y2, t = y1[:min_len], y2[:min_len], t[:min_len]
    if play_signals_on_load:
        print(f"Sampling Rate: {fs} Hz. Nyquist Limit: {fs / 2} Hz")

        # Play both original signals sequentially
        print("Playing y1 (Piano)...")
        utils.play_audio(y1, fs)
        print("Playing y2 (Flute)...")
        utils.play_audio(y2, fs)

    if show_spectre_of_loaded_signals:
        pa.plot_time_and_freq(t, y1, fs, title="y1 (392Hz)")
        pa.plot_time_and_freq(t, y2, fs, title="y2 (523Hz)")
    return t, fs, y1, y2

def load_2_signals(alternative = False):
    if alternative:
        return load_2_signals_no_recording()
    lin_space, fs, y2 = utils.load_signal('Drugi zadatak/y2.wav')

    y1 = utils.record_with_cache('y1_speech.wav', duration=2, fs=fs)

    samples_to_add = len(y2) - len(y1)
    y1_padded = np.append(y1, np.zeros(samples_to_add))

    # Now they are both 3 seconds long
    y1_padded = y1_padded / np.max(np.abs(y1_padded))
    y2 = y2 / np.max(np.abs(y2))
    return lin_space, fs, y1_padded * MAGIC_NUMBER, y2 * MAGIC_NUMBER


def zadatak_2_a():
    pass

def zadatak_2_b(should_show_graph = True):
    t, fs, y1, y2 = load_2_signals()
    if should_show_graph:
        pa.plot_time_and_freq(t, y1, fs, title="Originalni Signal y1 (Govor)")
        pa.plot_time_and_freq(t, y2, fs, title="Originalni Signal y2 (WAV File)")
    return t, fs, y1, y2

def zadatak_2_v(should_show_graph = True):
    t, fs, y1, y2 = zadatak_2_b(SHOW_ALL_STEPS)
    f_cutoff = F_CUTOFF
    y1_filtered = filters.lowpass_filter(y1, f_cutoff, fs, order=FILTER_ORDER)
    y2_filtered = filters.lowpass_filter(y2, f_cutoff, fs, order=FILTER_ORDER)
    if should_show_graph:
        pa.plot_time_and_freq(t, y1_filtered, fs, title="Filtrirani Signal y1n")
        pa.plot_time_and_freq(t, y2_filtered, fs, title="Filtrirani Signal y2n")

    if SHOULD_SAVE_FILES:
        utils.save(SAVE_DIR + "2v_y1", fs, y1_filtered)
        utils.save(SAVE_DIR + "2v_y2", fs, y2_filtered)
    return t, fs, y1_filtered, y2_filtered

def zadatak_2_g(should_show_graph = True):
    t, fs, y1, y2 = zadatak_2_v(SHOW_ALL_STEPS)
    y2m = y2 * np.cos(2 * np.pi * F_CARRIER * t)
    if should_show_graph:
        pa.plot_time_and_freq(t, y2m, fs, title="Modulisani Signal y2m")
    if SHOULD_SAVE_FILES:
        utils.save(SAVE_DIR + "2g_y2m", fs, y2m)

    return t, fs, y1, y2m

def get_transmission_signal():
    t, fs, y1, y2 = zadatak_2_g(SHOW_ALL_STEPS)
   # pa.plot_time_and_freq(t, y1, fs, title="AAAAAAAAAAAAAAAA y1")
   # pa.plot_time_and_freq(t, y2, fs, title = "AAAAAAAAAAAAAAAAAAAAAAAA y2")

    y1_filt = np.nan_to_num(y1, nan=0.0, posinf=0.0, neginf=0.0)
    y2_mod = np.nan_to_num(y2, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale by 0.5 so their sum (yt) never exceeds 1.0 (prevents clipping)
    yt = (y1_filt * 0.5) + (y2_mod * 0.5)

   # print(f"y1 has NaNs: {np.isnan(y1).any()}")
   # print(f"y2 has NaNs: {np.isnan(y2).any()}")
   # print(f"y1 max value: {np.max(np.abs(y1))}")
   # print(f"y2 max value: {np.max(np.abs(y2))}")

   # pa.plot_time_and_freq(t, yt, fs, title="y1 + y2 AAAAAAAAAAAAAAAAAAAAA")

  #  yt = y1 + y2
    return t, fs, yt

def zadatak_2_d():
    t, fs, yt = get_transmission_signal()
    pa.plot_time_and_freq(t, yt, fs, title="Poslati Signal yT")
    if SHOULD_SAVE_FILES:
        utils.save(SAVE_DIR + "2d_to_transm", fs, yt)
    return

def zadatak_2_dj(should_show_graph = True):
    t, fs, yt = get_transmission_signal()
    y_transmitted = filters.communication_channel(yt, fs, NYQUIST - 50, 6)
    if SHOULD_SAVE_FILES:
        utils.save(SAVE_DIR + "2dj_transmitted", fs, y_transmitted)
    if should_show_graph:
        pa.plot_time_and_freq(t, y_transmitted, fs, title="Primljen Signal yT")
    return t, fs, y_transmitted

def zadatak_2_e(should_show_graph = True):
    t, fs, y_received = zadatak_2_dj(SHOW_ALL_STEPS)
    #they overlap a bit oopsies
    y2_b = filters.bandpass_filter(y_received, BP_LOW, BP_HIGH, fs, order=FILTER_ORDER)

    if should_show_graph:
        pa.plot_time_and_freq(t, y2_b, fs, title="Rekonstruisan Signal y2_b")
    if SHOULD_SAVE_FILES:
        utils.save(SAVE_DIR + "2_e_y2_b", fs, y2_b)
    return t, fs, y_received, y2_b

def zadatak_2_zj(should_show_graph = True):
    t, fs, y1_recovered, y2_b = zadatak_2_e(SHOW_ALL_STEPS)

    y2_d = y2_b * np.cos(2 * np.pi * F_CARRIER * t)

    if should_show_graph:
        pa.plot_time_and_freq(t, y2_d, fs, title="Rekonstruisan Signal y2d")
    if SHOULD_SAVE_FILES:
        utils.save(SAVE_DIR + "2_zj_y2d", fs, y2_d)
    return t, fs, y1_recovered, y2_d

def zadatak_2_z(should_show_graph = True):
    t, fs, y_received, y2_d = zadatak_2_zj(SHOW_ALL_STEPS)
    y1_recovered = filters.lowpass_filter(y_received, F_CUTOFF, fs, order=FILTER_ORDER)
    y2_recovered = filters.lowpass_filter(y2_d, F_CUTOFF, fs, order=FILTER_ORDER)

    if should_show_graph:
        pa.plot_time_and_freq(t, y1_recovered, fs, title="Rekonstruisan Signal y1")
        pa.plot_time_and_freq(t, y2_recovered, fs, title="Rekonstruisan Signal y'")

    if SHOULD_SAVE_FILES:
        utils.save(SAVE_DIR + "2z_y1_recovered", fs, y1_recovered)
        utils.save(SAVE_DIR + "2z_y2_recovered", fs, y2_recovered)

    return t, fs, y1_recovered, y2_recovered


def zadatak_2_i():
    pass

def test_method():
    zadatak_2_z()
    return


