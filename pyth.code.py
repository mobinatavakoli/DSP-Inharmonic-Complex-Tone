import numpy as np
import scipy.signal as sig
import scipy.fftpack as fft
import librosa
from scipy.optimize import curve_fit
from scipy.signal import hilbert

def load_audio(path, sr=44100):
  
    x, fs = librosa.load(path, sr=sr, mono=True)
    return x, fs

def estimate_f0(x, fs, fmin=50, fmax=2000):
   
    f0 = librosa.yin(x, fmin=fmin, fmax=fmax, sr=fs)
   
    return np.nanmedian(f0)

def inharmonic_model(n, f0, B):
    
    return n * f0 * np.sqrt(1 + B * n**2)

def estimate_inharmonicity(x, fs, N_partials=10):
   
   
    X = np.abs(fft.fft(x))
    freqs = fft.fftfreq(len(x), 1/fs)
    pos_idx = freqs > 0
    X = X[pos_idx]
    freqs = freqs[pos_idx]
    
    
    peaks, _ = sig.find_peaks(X, height=np.max(X)*0.1, distance=20)
    peak_freqs = freqs[peaks]
    peak_mags = X[peaks]
    
    
    idx_sort = np.argsort(peak_mags)[::-1]
    peak_freqs = peak_freqs[idx_sort][:N_partials]
    
   
    f0 = estimate_f0(x, fs)
    
   
    n = np.arange(1, len(peak_freqs)+1)
    
    
    popt, _ = curve_fit(lambda nn, B: inharmonic_model(nn, f0, B),
                        n, peak_freqs, p0=[1e-4])
    B_est = popt[0]
    
    return f0, B_est, peak_freqs

def design_filter_bank(fs, partial_freqs, BW=20):
  
    filters = []
    for f_center in partial_freqs:
        low = max(0.1, (f_center - BW/2) / (fs/2))
        high = min(0.99, (f_center + BW/2) / (fs/2))
        b, a = sig.butter(4, [low, high], btype='bandpass')
        filters.append((b, a))
    return filters

def decompose_signal(x, filters):
   
    subbands = []
    for b, a in filters:
        y = sig.filtfilt(b, a, x)
        subbands.append(y)
    return subbands

def extract_envelopes(subbands):
   
    envelopes = []
    for y in subbands:
        analytic = hilbert(y)
        env = np.abs(analytic)
        envelopes.append(env)
    return envelopes

def reconstruct_signal(subbands):
 
    return np.sum(np.vstack(subbands), axis=0)



if __name__ == "__main__":
   
    x, fs = load_audio("tone_recording.wav", sr=44100)
    
    
    f0, B, part_freqs = estimate_inharmonicity(x, fs, N_partials=8)
    print(f"Estimated f0 = {f0:.2f} Hz, B = {B:.2e}")
    
   
    filters = design_filter_bank(fs, part_freqs, BW=30)
    
   
    subbands = decompose_signal(x, filters)
    
    
    envelopes = extract_envelopes(subbands)
    
   
    x_rec = reconstruct_signal(subbands)
    
   
    librosa.output.write_wav("reconstructed.wav", x_rec, fs)
    print("Done.")
