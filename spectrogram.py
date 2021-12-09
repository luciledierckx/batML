from skimage import filters
import numpy as np
from skimage.util.shape import view_as_windows
from scipy.ndimage import zoom


def denoise(spec_noisy, mask=None):
    """
    Performs denoising of the spectrogram by subtracting the mean from each frequency band.

    Parameters
    -----------
    spec_noisy : numpy array
        Noisy spectrogram.
    mask : numpy array
        Chooses the relevant time steps to use.
    
    Returns
    --------
    spec_denoise : numpy array
        Denoised spectrogram.
    """

    if mask is None:
        # no mask
        me = np.mean(spec_noisy, 1)
        spec_denoise = spec_noisy - me[:, np.newaxis]

    else:
        # user defined mask
        mask_inv = np.invert(mask)
        spec_denoise = spec_noisy.copy()

        if np.sum(mask) > 0:
            me = np.mean(spec_denoise[:, mask], 1)
            spec_denoise[:, mask] = spec_denoise[:, mask] - me[:, np.newaxis]

        if np.sum(mask_inv) > 0:
            me_inv = np.mean(spec_denoise[:, mask_inv], 1)
            spec_denoise[:, mask_inv] = spec_denoise[:, mask_inv] - me_inv[:, np.newaxis]

    # remove anything below 0
    spec_denoise.clip(min=0, out=spec_denoise)

    return spec_denoise


def gen_mag_spectrogram(x, fs, ms, overlap_perc):
    """
    Computes magnitude spectrogram by specifying the time.

    Parameters
    -----------
    x : numpy array
        Audio samples.
    fs : float
        Sampling rate
    ms : float
        Length of a Fast Fourier Transform window.
    overlap_perc :  float
        Percentage of overlap between windows.
    
    Returns
    --------
    spec : numpy array
        Magnitude spectrogram.
    x_win_len : int
        The number of windows in the audio samples.
    """

    nfft = int(ms*fs)
    noverlap = int(overlap_perc*nfft)

    # window data
    step = nfft - noverlap
    shape = (nfft, (x.shape[-1]-noverlap)//step) # (size of window, number of windows)
    strides = (x.strides[0], step*x.strides[0]) # (nbr of bytes to move from an element of the array to the other, nbr of bytes to move from a window to the other)
    x_wins = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # apply Hanning window (smoothing values)
    x_wins_han = np.hanning(x_wins.shape[0])[..., np.newaxis] * x_wins

    # do fft (rrft = discrete FT for real input)
    # note this will be much slower if x_wins_han.shape[0] is not a power of 2
    complex_spec = np.fft.rfft(x_wins_han, axis=0)

    # calculate magnitude (a+bj -> a^2 + b^2)
    mag_spec = (np.conjugate(complex_spec) * complex_spec).real

    # remove dc component and orient correctly
    spec = mag_spec[1:, :]
    spec = np.flipud(spec)
    
    return spec, x_wins.shape[0]


def gen_spectrogram(audio_samples, sampling_rate, fft_win_length, fft_overlap, crop_spec=True, max_freq=256, min_freq=0):
    """
    Computes the magnitude spectrogram, potentially crops it using a band-pass filter
    and computes the logarithm of the spectrogram.

    Parameters
    -----------
    audio_samples : numpy array
        Data read from wav file.
    sampling_rate : int
        Sample rate of wav file.
    fft_win_length : float
        Length of a Fast Fourier Transform window.
    fft_overlap :  float
        Percentage of overlap between windows.
    crop_spec : bool
        True to apply a band-pass filter to the spectrogram and False otherwise
    max_freq : int
        Index of the maximum frequency in the spectrogram array to do the band-pass filter.
    min_freq : int
        Index of the minimum frequency in the spectrogram array to do the band-pass filter.
    
    Returns
    --------
    spec : numpy array
        Log-magnitude spectrogram.
    """

    # compute spectrogram
    spec, x_win_len = gen_mag_spectrogram(audio_samples, sampling_rate, fft_win_length, fft_overlap)
    
    # only keep the relevant bands
    if crop_spec:
        freq = abs(np.fft.rfftfreq(x_win_len)*sampling_rate)
        freq = np.flip(freq)
        spec = spec[-max_freq:-min_freq, :]
        
        # add some zeros if spec too small
        req_height = max_freq-min_freq
        if spec.shape[0] < req_height:
            zero_pad = np.zeros((req_height-spec.shape[0], spec.shape[1]))
            spec = np.vstack((zero_pad, spec))

    # perform log scaling
    log_scaling = 2.0 * (1.0 / sampling_rate) * (1.0/(np.abs(np.hanning(int(fft_win_length*sampling_rate)))**2).sum())
    spec = np.log(1.0 + log_scaling*spec)

    return spec


def process_spectrogram(spec, denoise_spec=True, mean_log_mag=0.5, smooth_spec=True):
    """
    Denoises, and smooths spectrogram.

    Parameters
    -----------
    spec : numpy array
        Log-magnitude spectrogram.
    denoise_spec : bool
        True to denoise the spectrogram and False otherwise.
    mean_log_mag : float
        Minimum average log magnitude used as mask for denoising.
    smooth_spec : bool
        True to smooth the spectrogram and False otherwise.
    
    Returns
    --------
    spec : numpy array
        Potentially denoised and/or smoothed log-magnitude spectrogram.
    """

    # denoise
    if denoise_spec:
        # use a mask as there is silence at the start and end of recs
        mask = spec.mean(0) > mean_log_mag
        spec = denoise(spec, mask)

    # smooth the spectrogram
    if smooth_spec:
        spec = filters.gaussian(spec, 1.0)

    return spec


def compute_features_spectrogram(audio_samples, sampling_rate, params):
    """
    Computes overlapping windows of spectrogram as input for classifier.

    Parameters
    -----------
    audio_samples : numpy array
        Data read from wav file.
    sampling_rate : int
        Sample rate of wav file.
    params : DataSetParams
        Parameters of the model.
    
    Returns
    --------
    features : numpy array
        Array containing the spectrogram features for each window of the audio file.
    """

    # load audio and create log-magnitude spectrogram
    spectrogram = gen_spectrogram(audio_samples, sampling_rate, params.fft_win_length, params.fft_overlap,
                                     crop_spec=params.crop_spec, max_freq=params.max_freq, min_freq=params.min_freq)
    
    # denoise and smooth the spectrogram
    spectrogram = process_spectrogram(spectrogram, denoise_spec=params.denoise, mean_log_mag=params.mean_log_mag, smooth_spec=params.smooth_spec)

    # extract windows
    spec_win = view_as_windows(spectrogram, (spectrogram.shape[0], params.window_width))[0]
    
    spec_win = zoom(spec_win, (1, 0.5, 0.5), order=1) # (1, 0.5, 0.5) zoom factor for each of the 3 directions

    spec_width = spectrogram.shape[1]

    # make the correct size for CNN
    features = np.zeros((spec_width, 1, spec_win.shape[1], spec_win.shape[2]), dtype=np.float32)
    features[:spec_win.shape[0], 0, :, :] = spec_win

    return features
