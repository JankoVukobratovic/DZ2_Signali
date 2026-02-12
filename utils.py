import os
from typing import Tuple, Any

from numpy import ndarray, dtype, float64
from scipy.io import wavfile
from scipy.fftpack import fftshift, fftfreq
import sounddevice as sd
import scipy.signal as sci
import numpy as np


def record_with_cache(filename, duration, fs=44100):
    if os.path.exists(filename):
        #print(f"Loading cached recording from {filename} ---")
        fs_loaded, data = wavfile.read(filename)
        if fs_loaded != fs:
            pass
            print(f"Warning: Cached FS ({fs_loaded}) differs from requested FS ({fs}).")
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        return data
    else:
        print(f"--- No cache found. Starting NEW recording for {duration}s... ---")
        print("Press any key to continue")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        data = recording.flatten()
        save_data = (data * 32767).astype(np.int16)
        wavfile.write(filename, fs, save_data)
        print(f"--- Recording saved to {filename} ---")
        return data

def record(duration, fs=16000):
    print(f"Recording for {duration} seconds... speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Recording finished.")
    # Flatten to 1D array
    return recording.flatten()

def load_signal(file_name: str)-> tuple[ndarray[tuple[Any, ...], dtype[float64]], int, Any]:
    fs, data = wavfile.read(file_name)
    duration = len(data) / fs
    t = np.linspace(0, duration, len(data), endpoint=False)
    #print(f"fs:{fs}")
    return t, fs, data

def create_time_axis(duration, fs):
    return np.linspace(0, duration, int(fs * duration), endpoint=False)


def generate_sine(freq, t, amp=1.0, phase=0):
    return amp * np.sin(2 * np.pi * freq * t + phase)

def generate_sine_alternative(freq, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    return t, signal

def play_audio(signal, fs):
    sd.play(signal, fs)
    sd.wait()

