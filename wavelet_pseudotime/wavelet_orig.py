import numpy as np
import anndata as ad
import scipy.stats
from anndata import AnnData
from scipy.sparse import csr_matrix
import heapq
import pywt
from pywt import Wavelet
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from typing import Union, List, Tuple, Optional
from collections import defaultdict as dd

# TODO: The functions here should operate on the wavelet coefficient matrix rather than the pseudotime coefficient matrix


# TODO: move this to its own file?
class WaveletTransform:
    def __init__(self,
                 scales: list = None,
                 wavelet: Union[str, Wavelet] = "mexh"):
        self.scales = scales
        self.wavelet = wavelet
        return

    def apply(self,
              signal: np.array) -> (np.array, np.array):
        return pywt.cwt(signal, scales=self.scales, wavelet=self.wavelet)

    def reconstruct(self,
                    scale: float,
                    position: float,
                    length: int):
        return reconstruct_wavelet(scale, position, length, wavelet=self.wavelet)


def reconstruct_wavelet(scale: float,
                        position: float,
                        length: int,
                        wavelet: Union[str, Wavelet] = "mexh"):
    if isinstance(wavelet, str):
        wavelet = pywt.ContinuousWavelet(wavelet)

    x_values = np.linspace(0, length, num=length)
    wavelet_function = np.real(wavelet.wavefun(scale)[0])
    signal = np.zeros_like(x_values)
    start_point = position - len(wavelet_function) // 2
    for index, value in enumerate(wavelet_function):
        # Ensure we are within signal bounds
        if start_point + index < 0 or start_point + index >= len(signal):
            continue
        signal[start_point + index] = value
    return x_values, signal


def detect_local_maxima(arr: np.array) -> Tuple[np.array, np.array]:
    """
    Detects local maxima in a 2D array.
    Parameters
    ----------
    arr : np.array
        Array to process.

    Returns
    -------
    Tuple[np.array, np.array]
        X- and y-indices of local maxima.

    Notes
    -----
    There is a notable limitation to this method: maxima that are not exactly one-entry wide are not found.
    """
    # Comparing equal enforces more strict local maxima condition
    neighborhood = generate_binary_structure(len(arr.shape), 2)
    local_max = maximum_filter(arr, footprint=neighborhood) == arr
    background = (arr == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_maxima = local_max ^ eroded_background
    return np.where(detected_maxima)


def determine_scales_from_signal(pseudotime_signal: np.array,
                                 wavelet_transform: WaveletTransform = "mexh") -> np.array:
    """
    Determines the scales to use for the wavelet transform based on the number of samples in the input.
    Parameters
    ----------
    pseudotime_signal : np.array
        Array containing the signal of interest.
    wavelet : Union[str, Wavelet]
        Name of the wavelet being used, or a Wavelet object from pywt.
    Returns
    -------
    np.array
        Scales to use in the wavelet transform.
    """
    # Pseudotime isn't real time; we'll assume uniform sampling* from 0-1.
    # *(sampling isn't uniform, but I'm not dealing with that can of warms)
    max_freq = len(pseudotime_signal)/2  # Nyquist
    min_scale = pywt.frequency2scale(wavelet_transform.wavelet, max_freq)
    min_power = int(min_scale)
    max_power = 4  # scale=16 corresponds to very low freq signal; seems unlikely to be informative below that
    return [2**i for i in range(min_power, max_power)]


def identify_max_scaletime(pseudotime_signal: np.array,
                           wavelet_transform: WaveletTransform,
                           coef_threshold: float = -np.inf) -> List[tuple]:
    """
    Returns the scale, time locations of the max wavelet coefficients.
    Parameters
    ----------
    pseudotime_signal : np.array
        Array containing the signal of interest.
    wavelet_transform : WaveletTransform
        WaveletTransform to use for computing coefficients.
    coef_threshold : float
        Minimum value for the coefficient to be considered a valid maximum.

    Returns
    -------
    dict
        Pair containing the scale and time locations of the max wavelet coefficients
    """
    coefs, freqs = wavelet_transform.apply(pseudotime_signal)
    max_scale, max_t = detect_local_maxima(np.abs(coefs))
    max_pairs = []
    for s, t in zip(max_scale, max_t):
        if np.abs(coefs[s, t]) >= coef_threshold:
            max_pairs.append((int(s), int(t)))
    return max_pairs


def determine_coef_threshold(pseudotime_signal: np.array,
                             wavelet_transform: WaveletTransform,
                             num_scrambles: int = 100,
                             pval_threshold: float = 0.05) -> float:
    """
    Obtains a minimum meaningful threshold by phase-scrambling the signal, obtaining the wavelet transform, and
    repeating until it is able to generate a null distribution
    Parameters
    ----------
    pseudotime_signal : np.array
        Signal of interest.
    wavelet : Union[WaveletTransform, str, Wavelet]
        Wavelet transform to use
    scales : List[float]
        Scales to use for the cwt. Ignored if `wavelet` is a WaveletTransform.
    num_scrambles : int
        The number of times to randomly phase-scramble the signal.
    pval_threshold : float
        The upper limit for the p-value.
    Returns
    -------
    float
        Threshold for the coefficient value to have p <= pval_threshold.
    """
    signal_fft = np.fft.fft(pseudotime_signal)
    all_coefs = []

    for _ in range(num_scrambles):
        phase_randomizer = np.exp(1j * (2*np.pi*np.random.random(len(pseudotime_signal))-np.pi))  # random phase
        scrambled_signal = np.fft.ifft(signal_fft * phase_randomizer).real  # invert phase-scrambled signal; keep real signal in case of numerical error
        coefs, _ = wavelet_transform.apply(scrambled_signal)  # get wavelet coefficients
        all_coefs += [*np.abs(coefs).ravel()]  # take absolute value; return
    all_coefs.sort()
    min_coef_idx = np.ceil(len(all_coefs) * (1-pval_threshold))
    return all_coefs[min_coef_idx]


def get_coef_pvalues(pseudotime_signal: np.array,
                     wavelet_transform: WaveletTransform,
                     num_scrambles: int = 0):
    """
    Obtains a minimum meaningful threshold by phase-scrambling the signal, obtaining the wavelet transform, and
    repeating until it is able to generate a null distribution
    Parameters
    ----------
    pseudotime_signal : np.array
        Signal to analyze.
    wavelet_transform : WaveletTransform
        The transform to use for getting the coefs.
    num_scrambles : int
        Number of times to scramble the phase to generate the null distribution.

    Returns
    -------

    """
    coefficients = []

    # Get the FFT of pseudotime_signal outside the loop
    signal_fft = np.fft.rfft(pseudotime_signal)

    for _ in range(num_scrambles):
        # Randomize the phase
        random_phase = np.exp(1j * (np.random.uniform(0, np.pi, signal_fft.shape)))
        scrambled_signal = signal_fft * random_phase

        # Reconstruct the signal
        reconstructed_signal = np.fft.irfft(scrambled_signal)

        # Compute the wavelet coefficients
        coefficients_current, _ = wavelet_transform.apply(reconstructed_signal)
        coefficients.extend(coefficients_current.ravel())

    # Sort all coefficients
    coefficients = sorted(np.abs(coefficients))

    # Get only local maxima of coefficients
    signal_coefs, _ = wavelet_transform.apply(pseudotime_signal)
    max_scaletimes = identify_max_scaletime(pseudotime_signal, wavelet_transform=wavelet_transform)  # TODO: re-running CWT here
    local_max_coefs = [signal_coefs[i,j] for (i, j) in max_scaletimes]
    real_coefficients = sorted(np.abs(local_max_coefs).ravel())
    p_map = {}
    idx = 0
    for coef in real_coefficients:
        where_coef = np.where(coefficients[idx:] > coef)
        if len(where_coef[0]) == 0:
            continue
        where_coef = where_coef[0][0]
        p = 1 - ((where_coef + idx) / len(coefficients))
        idx += where_coef
        p_map[np.round(coef, 12)] = p
    p_array = np.ones(signal_coefs.shape)
    for i in range(signal_coefs.shape[0]):
        for j in range(signal_coefs.shape[1]):
            p_key = round(np.abs(signal_coefs[i,j]), 12)
            if p_key in p_map:
                p_array[i,j] = p_map[np.round(np.abs(signal_coefs[i,j]), 12)]
    return signal_coefs, p_array


def get_significant_wavelets(pseudotime_signals: dict,
                             wavelet_transform: WaveletTransform,
                             fdr_rate: float = 0.05):
    """
    Gets the significant wavelets for a series of pseudotime signals and performs BH correction.
    Parameters
    ----------
    pseudotime_signals : dict
        Gene expression signals in pseudotime, keyed by gene.
    wavelet_transform : WaveletTransform
        Transform to use for getting the wavelet coefficients
    fdr_rate : float, optional
        Group-wise FDR value.

    Returns
    -------

    Notes
    -----
    This function takes a long time and does not identify wavelets that are much different from the much-faster
    `get_max_wavelets` function.

    """
    # First compute p values for all signals
    coef_dict = {}
    pvalue_dict = {}
    all_p_values = []
    for signal_name, signal_data in pseudotime_signals.items():
        coefs, p_values = get_coef_pvalues(signal_data, wavelet_transform)
        coef_dict[signal_name] = coefs
        pvalue_dict[signal_name] = p_values
        all_p_values.extend(p_values[p_values < 1])  # 1 is the default value
    all_p_values = np.array(all_p_values)
    corrected_p = scipy.stats.false_discovery_control(all_p_values)
    p_thresh = np.max(all_p_values[corrected_p < fdr_rate])

    # Group significant wavelets by time & scale
    significant_wavelets = dd(set)
    significant_coefs = dd(float)
    for gene, pvalues in pvalue_dict.items():
        i, j = np.where(pvalues <= p_thresh)
        for ii, jj in zip(i, j):
            significant_wavelets[ii, jj].add(gene)
            significant_coefs[gene, ii, jj] = coef_dict[gene][ii, jj]
    return significant_wavelets, significant_coefs


def get_max_wavelets(pseudotime_signals: dict,
                     wavelet_transform: WaveletTransform,
                     normalize_signal: bool = True,
                     top_fraction: float = None):
    """

    Parameters
    ----------
    pseudotime_signals : dict
        Dict of signals, keyed by gene name, containing gene expression along pseudotime.
    wavelet_transform : WaveletTransform
        Wavelet transform to use.
    normalize_signal : bool, optional
        Whether to normalize the signals before getting their wavelet coefficients.
    top_fraction : float
        Fraction of coefficients to retain, valued 0-1. 0.05 would retain the top 5% largest coefficients. If None,
        returns all coefficients.

    Returns
    -------
    defaultdict
        dict of sets containing the gene names, keyed by [scale, position] of the wavelet.
    tuple
        Group of three aligned lists. tuple[0] has the values of the coefficients. tuple[1] has the name of the
        associated gene. tuple[2] has the scale and time of the wavelet.
    """
    # TODO: Normalize by zero-mean and std instead of 0-1
    # Initialize
    coef_dict = {}
    significant_wavelets = dd(set)
    all_max_coefs = []
    all_max_genes = []
    all_max_scaletime = []
    for signal_name, signal_data in pseudotime_signals.items():
        if normalize_signal:
            mmin = np.min(signal_data)
            mmax = np.max(signal_data)
            signal_data = (signal_data - mmin) / (mmax-mmin)  # normalize to 0-1
        coefs, freqs = wavelet_transform.apply(signal_data)  # Get wavelet coefficients
        coef_x, coef_y = detect_local_maxima(coefs)  # Get the 2D local maxima among the coef matrix
        for i, j in zip(coef_x, coef_y):
            significant_wavelets[wavelet_transform.scales[i], j].add(signal_name)
            if top_fraction is not None:
                all_max_coefs.append(coefs[i,j])
                all_max_genes.append(signal_name)
                all_max_scaletime.append((i,j))
        coef_dict[signal_name] = coefs

    if top_fraction is not None:
        top_indices = heapq.nlargest(int(len(all_max_coefs) * top_fraction), range(len(all_max_coefs)), key=all_max_coefs.__getitem__)
        all_max_coefs = [all_max_coefs[i] for i in top_indices]
        all_max_genes = [all_max_genes[i] for i in top_indices]
        all_max_scaletime = [all_max_scaletime[i] for i in top_indices]

    return significant_wavelets, (all_max_coefs, all_max_genes, all_max_scaletime)