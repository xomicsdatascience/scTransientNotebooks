from pywt import Wavelet
from typing import Union, List, Tuple
import numpy as np
import pywt
from collections import defaultdict as dd, deque
import heapq
import pandas as pd


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

    def reconstruct_and_fit(self,
                            scale: float,
                            position: float,
                            length: int,
                            signal: np.array):
        pos, r = reconstruct_wavelet(scale, position, length, wavelet=self.wavelet)
        r = np.column_stack((r, np.ones_like(r)))
        coefs = np.linalg.lstsq(r, signal, rcond=None)[0]
        return pos, r@coefs


def reconstruct_wavelet(scale: float,
                        position: float,
                        length: int,
                        wavelet: Union[Wavelet, str] = "mexh"):
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





def find_local_extrema(arr: np.ndarray,
                       min_or_max: str = None) -> List[Tuple[int, int, str]]:
    """
    Find local extrema in a 2D array.

    A local extremum is defined as a connected group (8-connected)
    of pixels with the same value such that all adjacent pixels (that are not
    part of the group) are either strictly lower (for a maximum) or strictly higher
    (for a minimum) than the group.

    In the event that multiple pixels form the extremum, the returned coordinate is
    the center of the grouping (computed as the rounded average of their coordinates).

    Parameters
    ----------
        arr (np.ndarray): np.array

    Returns:
        List[Tuple[int, int, str]]
            A list of tuples with (row, column, type) where type is either "maximum" or "minimum".
    """
    # Dimensions of the array
    rows, cols = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    extrema = []

    # Define neighbors relative positions for 8-connectedness
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]

    def flood_fill(r: int, c: int) -> List[Tuple[int, int]]:
        """Flood fill to get all connected pixels with the same value."""
        group = []
        value = arr[r, c]
        queue = deque()
        queue.append((r, c))
        visited[r, c] = True

        while queue:
            cr, cc = queue.popleft()
            group.append((cr, cc))
            for dr, dc in neighbors:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                    if arr[nr, nc] == value:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        return group

    # Iterate over each element to identify groups (plateaus)
    for r in range(rows):
        for c in range(cols):
            if not visited[r, c]:
                group = flood_fill(r, c)
                # Determine the set of neighbor coordinates outside group
                group_set = set(group)
                neighbor_values = []

                for gr, gc in group:
                    for dr, dc in neighbors:
                        nr, nc = gr + dr, gc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in group_set:
                            neighbor_values.append(arr[nr, nc])

                # If there are no neighbor pixels (which can happen for an isolated pixel
                # in a 1x1 array), we cannot make a decision, so we continue.
                if not neighbor_values:
                    continue

                current_value = arr[r, c]
                is_max = all(neighbor < current_value for neighbor in neighbor_values)
                is_min = all(neighbor > current_value for neighbor in neighbor_values)

                # If the plateau qualifies as an extremum, calculate its center coordinate.
                if is_max or is_min:
                    # Compute center as the average of the group's row and col indices.
                    rows_group, cols_group = zip(*group)
                    center_r = int(round(sum(rows_group) / len(rows_group)))
                    center_c = int(round(sum(cols_group) / len(cols_group)))
                    extremum_type = "maximum" if is_max else "minimum"
                    if min_or_max is None:
                        extrema.append((center_r, center_c, extremum_type))
                    elif min_or_max == "max" and extremum_type == "maximum":
                        extrema.append((center_r, center_c, extremum_type))
                    elif min_or_max == "min" and extremum_type == "min":
                        extrema.append((center_r, center_c, extremum_type))
    return extrema


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
    # Initialize
    significant_wavelets = dd(set)
    all_max_coefs = []
    all_max_genes = []
    all_max_scaletime = []
    for signal_name, signal_data in pseudotime_signals.items():
        if normalize_signal:
            mmean = np.mean(signal_data)
            sstd = np.std(signal_data)
            signal_data = (signal_data - mmean) / sstd

        coefs, freqs = wavelet_transform.apply(signal_data)  # Get wavelet coefficients
        max_scaletimes = find_local_extrema(coefs, min_or_max="max")  # Get the 2D local maxima among the coef matrix

        for scale, time, _ in max_scaletimes:
            significant_wavelets[wavelet_transform.scales[scale], time].add(signal_name)
            if top_fraction is not None:
                all_max_coefs.append(coefs[scale, time])
                all_max_genes.append(signal_name)
                all_max_scaletime.append((scale, time))

    if top_fraction is not None:
        top_indices = heapq.nlargest(int(len(all_max_coefs) * top_fraction), range(len(all_max_coefs)), key=all_max_coefs.__getitem__)
        all_max_coefs = [all_max_coefs[i] for i in top_indices]
        all_max_genes = [all_max_genes[i] for i in top_indices]
        all_max_scaletime = [all_max_scaletime[i] for i in top_indices]

    return significant_wavelets, (all_max_coefs, all_max_genes, all_max_scaletime)


def get_max_scored_wavelets(pseudotime_signals: dict,
                            wavelet_transform: WaveletTransform,
                            scoring_function: callable,
                            score_threshold: float,
                            normalize_signal: bool = True,
                            top_fraction: float = None):
    """

    Parameters
    ----------
    pseudotime_signals : dict
        Dict of signals, keyed by gene name, containing gene expression along pseudotime.
    wavelet_transform : WaveletTransform
        Wavelet transform to use.
    scoring_function : callable
        Scoring function to use.
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
    # Initialize
    significant_wavelets = dd(set)
    all_max_coefs = []
    all_max_genes = []
    all_max_scaletime = []
    all_max_scores = []
    for signal_name, signal_data in pseudotime_signals.items():
        if normalize_signal:
            mmean = np.mean(signal_data)
            sstd = np.std(signal_data)
            signal_data = (signal_data - mmean) / sstd

        coefs, freqs = wavelet_transform.apply(signal_data)  # Get wavelet coefficients
        scores = scoring_function(coefs)  # Get the 2D local maxima among the coef matrix
        if scores is None:
            continue
        scaletimes = np.where(scores > score_threshold)
        for scale, time in zip(scaletimes[0], scaletimes[1]):
            significant_wavelets[wavelet_transform.scales[scale], time].add(signal_name)
            if top_fraction is not None:
                all_max_scores.append(scores[scale, time])
                all_max_coefs.append(coefs[scale, time])
                all_max_genes.append(signal_name)
                all_max_scaletime.append((scale, time))

    if top_fraction is not None:
        # top_indices = heapq.nlargest(int(len(all_max_coefs) * top_fraction), range(len(all_max_coefs)), key=all_max_coefs.__getitem__)
        top_indices = heapq.nlargest(int(len(all_max_coefs) * top_fraction), range(len(all_max_scores)),
                                     key=all_max_scores.__getitem__)
        all_max_coefs = [all_max_coefs[i] for i in top_indices]
        all_max_genes = [all_max_genes[i] for i in top_indices]
        all_max_scaletime = [all_max_scaletime[i] for i in top_indices]
    # print(f"all_max_genes: {all_max_genes}")
    return significant_wavelets, (all_max_coefs, all_max_genes, all_max_scaletime)


def get_scored_wavelet_across_sigs(pseudotime_signals: dict,
                            wavelet_transform: WaveletTransform,
                            scoring_function: callable,
                            score_threshold: float,
                            normalize_signal: bool = True,
                            top_fraction: float = None):
    """

    Parameters
    ----------
    pseudotime_signals : dict
        Dict of signals, keyed by gene name, containing gene expression along pseudotime.
    wavelet_transform : WaveletTransform
        Wavelet transform to use.
    scoring_function : callable
        Scoring function to use.
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
    # Initialize
    significant_wavelets = dd(set)
    scores_dict = {}
    scores_wavelet = {}
    for signal_name, signal_data in pseudotime_signals.items():
        if normalize_signal:
            mmean = np.mean(signal_data)
            sstd = np.std(signal_data)
            signal_data = (signal_data - mmean) / sstd

        coefs, freqs = wavelet_transform.apply(signal_data)  # Get wavelet coefficients
        score, st = scoring_function(coefs, get_st=True)  # Get the 2D local maxima among the coef matrix
        scores_dict[signal_name] = score
        scores_wavelet[signal_name] = st

    genes = [k for k, v in scores_dict.items() if v > score_threshold]
    for g in genes:
        significant_wavelets[g] = scores_wavelet[g]
    return significant_wavelets, scores_dict


def std_from_median(coefs):
    """Compute the number of standard deviations from the median of the wavelet coefficients."""
    s = np.std(coefs)
    if s > 0:
        return (coefs - np.median(coefs)) / s
    else:
        return None

def mag_median(coefs, get_st: bool = False):
    zmod = std_from_median(coefs)
    if zmod is None:
        if not get_st:
            return 0
        else:
            return 0, [None, None]
    else:
        scale, time = np.unravel_index(np.argmax(zmod), zmod.shape)
        if not get_st:
            return np.abs(coefs[scale,time] * zmod[scale,time])
        else:
            return np.abs(coefs[scale,time] * zmod[scale,time]), [scale, time]


def link_two_pt_paths(df1,
                      df2,
                      df1_pt_col: str = "dpt_pseudotime",
                      df2_pt_col: str = "dpt_pseudotime",
                      newcol: str = "stitched_pseudotime"):
    # Get overlapping entries
    inner = pd.merge(df1, df2, how="inner", suffixes=("_1", "_2"), left_index=True, right_index=True)
    # Adjust to the same spread, then take the average of their old and new values
    if df1_pt_col == df2_pt_col:
        inner_df1_pt_col = df1_pt_col + "_1"
        inner_df2_pt_col = df2_pt_col + "_2"
    orig_min = np.min(inner[inner_df1_pt_col])
    orig_max = np.max(inner.loc[inner[inner_df1_pt_col] != np.inf, inner_df1_pt_col])
    orig_mean = np.mean(inner[inner_df1_pt_col])
    # print(inner.shape)
    # return inner
    # print(orig_mean)
    orig_std = np.std(inner[inner_df1_pt_col])
    df1_dist = (inner[inner_df1_pt_col].values - orig_min) / (orig_max - orig_min)
    df2_dist = (inner[inner_df2_pt_col].values - np.min(inner[inner_df2_pt_col])) / (np.max(inner[inner_df2_pt_col]) - np.min(inner[inner_df2_pt_col]))
    mean_dist = (df1_dist + df2_dist) / 2
    # return mean_dist
    # Map center to old dist
    df2_dist = mean_dist * (orig_max - orig_min) + orig_min
    # centroid = np.mean(mean_dist[(mean_dist != np.inf) & ~(np.isnan(mean_dist)) ])
    # print(centroid)
    df2_cp = df2.copy()
    # df2_cp[df2_pt_col] = df2[df2_pt_col] * (orig_max-orig_min) + orig_min
    # df2_cp[df2_pt_col] += centroid
    all_idx = np.unique(np.concatenate((df1.index.values, df2.index.values)))
    df_new = pd.DataFrame(index=all_idx)
    df_new.loc[list(df1.index), newcol] = df1[df1_pt_col].values
    df_new.loc[list(df2.index), newcol] = df2_cp[df2_pt_col].values + np.mean(df1.loc[inner.index, df1_pt_col])
    new_max = np.max(df_new.loc[df_new[newcol] != np.inf, newcol])
    # return df_new
    df_new[newcol] = (df_new[newcol] - np.min(df_new[newcol])) / (new_max - np.min(df_new[newcol]))
    return df_new