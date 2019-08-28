import librosa
import numpy as np


def compute_melgram(audio_path, SR=12000, N_FFT=512, N_MELS=96, HOP_LEN=256, DURA=29.12):
    ''' Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366), where
    96 == #mel-bins and 1366 == #time frame

    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.
    More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..
    '''

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) // 2]
    logam = librosa.power_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS) ** 2,
                ref=1.0)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret


def compute_melgram_multi_slice(audio_path, SR=12000, N_FFT=512, N_MELS=96, HOP_LEN=256, DURA=29.12):
    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)

    srcs = []

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
        srcs.append(src)
    else:
        slice_num = n_sample // n_sample_fit
        for i in range(slice_num):
            temp = src[i * n_sample_fit:(i + 1) * n_sample_fit]
            srcs.append(temp)
        if n_sample % n_sample_fit > 0.5 * n_sample_fit:
            srcs.append(src[-1 * n_sample_fit:])
    rets = []
    for src in srcs:
        logam = librosa.power_to_db
        melgram = librosa.feature.melspectrogram
        ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                            n_fft=N_FFT, n_mels=N_MELS) ** 2,
                    ref=1.0)
        ret = ret[np.newaxis, np.newaxis, :]
        rets.append(ret)
    return rets


if __name__ == '__main__':
    audio_path = "../audio/Fabel-不生不死.mp3"
    mel_spectrum = compute_melgram_multi_slice(audio_path, SR=22050, N_FFT=2048, N_MELS=128, HOP_LEN=1024, DURA=30)
    print(mel_spectrum.shape)
