import numpy as np
from anndata import AnnData
from typing import Union, List, Tuple
from .windowing import Window, GaussianWindow, RectWindow, ConfinedGaussianWindow
from collections import defaultdict as dd
from .wavelets import WaveletTransform, get_max_wavelets, get_max_scored_wavelets, std_from_median, get_scored_wavelet_across_sigs, mag_median
from scipy.optimize import curve_fit
from sklearn.metrics import explained_variance_score
from typing import Literal
import scanpy as sc
import wavelet_pseudotime
# from wavelet_pseudotime.windowing import
from wavelet_pseudotime.binning import quantile_binning_anndata
from wavelet_pseudotime.load_data import load_astral, gene_coverage
from pypsupertime import Psupertime
import pypsupertime
import os
# from wavelet_pseudotime.load_data import gene_coverage

def window_trajectory(adata: AnnData,
                      trajectory: list,
                      normalize_to_endtime: bool = True,
                      window: Union[str, Window] = "gaussian",
                      window_params: dict = None,
                      pseudotime_col: str = "dpt_pseudotime",
                      node_col: str = "leiden",
                      exclude_minimum: bool = False,
                      exclude_ends: int = None):
    # initialize window
    window = _init_window(window, window_params)

    # Get pseudotime positions; limit to nodes in trajectory
    if trajectory is not None:
        retain_idx = adata.obs[node_col].isin(trajectory)
    else:
        retain_idx = np.ones(adata.shape[0], dtype=bool)
    pseudotime = adata.obs.loc[retain_idx, pseudotime_col]

    if normalize_to_endtime:
        pseudotime = _normalize_to_endtime(pseudotime)
    # print(sum(retain_idx))
    if exclude_ends is None:
        return window.apply(positions=pseudotime,
                        values=adata.X[retain_idx, :],
                        exclude_minimum=exclude_minimum)

    pseudotime_signals = window.apply(positions=pseudotime,
                        values=adata.X[retain_idx, :],
                        exclude_minimum=exclude_minimum)
    return pseudotime_signals[exclude_ends:-exclude_ends, :]

def _normalize_to_endtime(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def _init_window(window: Union[str, Window],
                 window_params: dict = None):
    if issubclass(window.__class__, str):
        if window.lower() == "gaussian":
            if window_params is None:
                window_params = {"n_windows": 25, "sigma": 0.2}
            window = GaussianWindow(**window_params)
        elif window.lower() == "rect":
            if window_params is None:
                window_params = {"n_windows": 25, "width": 0.1}
            window = RectWindow(**window_params)
        else:
            raise ValueError(f"Unrecognized window type: {window}")
    return window


def extract_by_scale(best_tuple):
    wavelet_dict = dd(list)
    num_entries = len(best_tuple[0])
    for i in range(num_entries):
        scale_idx = best_tuple[2][i][0]
        wavelet_dict[scale_idx].append((best_tuple[0][i], best_tuple[1][i], best_tuple[2][i][1]))
    return wavelet_dict

def extract_by_scaletime(coefs, genes, scaletime):
    wavelet_dict = dd(list)
    num_entries = len(coefs)
    for i in range(num_entries):
        scale_idx = scaletime[i][0]
        time_idx = scaletime[i][1]
        wavelet_dict[scale_idx, time_idx].append(genes[i])
    return wavelet_dict

def extract_by_gene(coefs, genes, scaletime):
    wavelet_dict = dd(list)
    num_entries = len(coefs)
    for i in range(num_entries):
        wavelet_dict[genes[i]].append(scaletime[i])
    return wavelet_dict


def extract_genecount(coefs: list,
                         genes: list,
                         scaletime: list):
    num_entries = len(coefs)
    (min_scale, max_scale), max_time = get_scaletime_bounds(scaletime)
    coef_counts = np.zeros((max_scale+1, max_time+1))
    for i in range(num_entries):
        coef_counts[scaletime[i]] += 1
    return coef_counts

def extract_coef_weights(coefs: list,
                         genes: list,
                         scaletime: list):
    num_entries = len(coefs)
    (min_scale, max_scale), max_time = get_scaletime_bounds(scaletime)
    coef_weights = np.zeros((max_scale+1, max_time+1))
    for i in range(num_entries):
        coef_weights[scaletime[i]] += np.abs(coefs[i])

    return coef_weights

def extract_genepower(coefs: list,
                      genes: list,
                      scaletime: list,
                      pseudotime_signals: dict,
                      wavelet_transform: WaveletTransform):
    num_entries = len(genes)
    (min_scale, max_scale), max_time = get_scaletime_bounds(scaletime)
    genepower = np.zeros((max_scale+1, max_time+1))
    for i in range(num_entries):
        gene = genes[i]
        power = pseudotime_signals[gene].T @ pseudotime_signals[gene]
        genepower[scaletime[i][0], scaletime[i][1]] += power
    gene_count = extract_genecount(coefs, genes, scaletime)
    return genepower/gene_count


def extract_var_explained(coefs: list,
                          genes: list,
                          scaletime: list,
                          pseudotime_signals: dict,
                          wavelet_transform: WaveletTransform):
    num_entries = len(coefs)
    (min_scale, max_scale), max_time = get_scaletime_bounds(scaletime)
    var_explained = np.zeros((max_scale+1, max_time+1))
    var_explained_dict = {}
    for i in range(num_entries):
        scale = wavelet_transform.scales[scaletime[i][0]]
        gene = genes[i]
        _, r = wavelet_transform.reconstruct(scale=scale,
                                             position=scaletime[i][1],
                                             length=len(pseudotime_signals[gene]))
        cf, _, _, _ = np.linalg.lstsq(np.column_stack([r, np.ones(len(r))]), pseudotime_signals[gene], rcond=None)
        # (a, b), pcov = curve_fit(linear_function, r, pseudotime_signals[gene])
        # r = linear_function(r, a, b)
        # var_exp = explained_variance_score(pseudotime_signals[gene], r)
        var_exp = explained_variance_score(pseudotime_signals[gene], np.column_stack([r, np.ones(len(r))]) @ cf)
        var_explained[scaletime[i][0], scaletime[i][1]] += var_exp
        var_explained_dict[gene] = var_exp
    gene_count = extract_genecount(coefs, genes, scaletime)
    return var_explained / gene_count


def extract_var_explained_by_allwavelets(coefs: list,
                                          genes: list,
                                          scaletime: list,
                                          pseudotime_signals: dict,
                                          wavelet_transform: WaveletTransform):
    wavelets = gather_wavelets_by_gene(genes, scaletime)
    var_explained = {}
    for gene, st in wavelets.items():
        wavelet_sigs = []
        for s, t in st:
            _, r = wavelet_transform.reconstruct(scale=wavelet_transform.scales[s],
                                                 position=t,
                                                 length=len(pseudotime_signals[gene]))
            wavelet_sigs.append(r)
        wavelet_sigs = np.column_stack(wavelet_sigs)
        coefs = np.linalg.lstsq(wavelet_sigs,
                                         pseudotime_signals[gene],
                                         rcond=None)[0]
        var_explained[gene] = explained_variance_score(pseudotime_signals[gene], coefs @ wavelet_sigs)
    # Return keys sorted by value in descending order
    return sorted(var_explained.keys(), key=lambda k: var_explained[k], reverse=True)


def gather_wavelets_by_gene(genes: list,
                            scaletime: list):
    wavelets_by_gene = dd(list)
    for g, st in zip(genes, scaletime):
        wavelets_by_gene[g].append(st)
    return wavelets_by_gene


def linear_function(x, a, b):
    return a * x + b


def get_scaletime_bounds(scaletime: List[Tuple[int, int]]):
    min_scale = np.inf
    max_scale = -np.inf
    max_time = -np.inf
    for s, t in scaletime:
        if s > max_scale:
            max_scale = s
        if s < min_scale:
            min_scale = s
        if t > max_time:
            max_time = t
    return (min_scale, max_scale), max_time

# num_entries = len(best_genes[0])
# best_coefs_counts = np.zeros((len(w.scales), 25))
# best_coefs_weight = np.zeros((len(w.scales), 25))
# best_coefs_genepower = np.zeros((len(w.scales), 25))
# var_explained = np.zeros((len(w.scales), 25))
# idx_map = {}
# for idx, s in enumerate(w.scales):
#     idx_map[s] = idx
# for i in range(num_entries):
#     idx = best_genes[2][i]
#     best_coefs_counts[idx[0], idx[1]] += 1
#     best_coefs_weight[idx[0], idx[1]] += np.abs(best_genes[0][i])
#
#     gene = best_genes[1][i]
#     # best_coefs_genepower[idx[0], idx[1]] += calc_power(sigs[gene])
#     # s = sigs[gene] - np.mean(sigs[gene])
#     best_coefs_genepower[idx[0], idx[1]] += sigs[gene].T @ sigs[gene]
#     _, r = wavelet_pseudotime.wavelet.reconstruct_wavelet(scale=w.scales[idx[0]], position=idx[1], length=25,
#                                                           wavelet="mexh")
#     popt, pcov = curve_fit(linear_func, r, sigs[gene])
#
#     # Extract the best-fit parameters
#     a_opt, b_opt = popt
#     r = linear_func(r, a_opt, b_opt)
#     var_exp = explained_variance_score(sigs[best_genes[1][i]], r)
#     var_explained[idx[0], idx[1]] = var_exp
# best_coefs_genepower /= best_coefs_counts
# var_explained /= best_coefs_counts

def pipeline(adata: AnnData,
             preprocess: bool = True,
             pseudotime_col: str = None,
             path: list = None,
             binning_method: Literal["window", "quantile"] = "window",
             scoring_function: callable = None,
             scoring_threshold: float = None,
             save_dir: str = None):
    """
    Processes an AnnData object to detect transient changes in expression.
    Parameters
    ----------
    adata
    preprocess
    pseudotime_col
    path
    binning_method
    scoring_function
    save_dir

    Returns
    -------

    """
    # Parameters; TODO: add to arguments, or otherwise make accessible
    num_bins = 50
    sigma = 0.03
    max_distance = 3*sigma
    node_col = "leiden"
    top_fraction = 0.01

    # Data is loaded; do preprocess
    if "iroot" not in adata.uns and (pseudotime_col is None or pseudotime_col not in adata.obs.columns):
        raise (ValueError("Pseudotime needs to be defined in the input data or 'adata.uns['iroot']' needs to be set."))
    if preprocess:
        print("Preprocessing... ", end="", flush=True)
        sc.pp.normalize_total(adata)  # Normalize to some total.
        sc.pp.log1p(adata)  # Take the natural log of the data.
        sc.pp.scale(adata)  # rescale to 0 mean and variance of 1
        # TODO: add batch correction, regress_out
        sc.pp.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=40, n_pcs=20)

        sc.tl.leiden(adata, resolution=1)
        sc.tl.diffmap(adata)
        if "iroot" in adata.uns:
            print("\rComputing pseudotime...", end="", flush=True)
            sc.tl.dpt(adata)  # needs root node

        sc.tl.paga(adata)
        sc.pl.paga(adata, show=False)
        sc.tl.umap(adata, init_pos='paga')

    # Do gene expression along PT
    print("\rComputing gene expression along pseudotime...", end="", flush=True)
    if binning_method == "quantile":
        pseudotime_signals = quantile_binning_anndata(adata,
                                                      pseudotime_key=pseudotime_col,
                                                      num_bins=num_bins)
        pseudotime_signals.drop(columns="_xbins", inplace=True)
    elif binning_method == "window":
        window = ConfinedGaussianWindow(n_windows=num_bins,
                                        sigma=sigma,
                                        max_distance=max_distance)

        pseudotime_signals = window_trajectory(adata,
                                               trajectory=path,
                                               node_col=node_col,
                                               window=window,
                                               pseudotime_col=pseudotime_col,
                                               exclude_minimum=False)
    else:
        raise(NotImplementedError(f"Binning method {binning_method} not supported."))

    # Get wavelet transform + score
    print("\rComputing wavelet transform...", end="", flush=True)
    wt = WaveletTransform(scales=np.arange(1, 7), wavelet="mexh")

    if scoring_function is None:
        waves, (coefs, genes, scaletime) = get_max_wavelets(pseudotime_signals,
                                                            wt,
                                                            top_fraction=top_fraction)
    else:
        if scoring_threshold is None:
            raise(ValueError("A threshold must be specified with the scoring function."))
        waves, (coefs, genes, scaletime) = get_max_scored_wavelets(pseudotime_signals,
                                                                   wt,
                                                                   scoring_function=scoring_function,
                                                                   scoring_threshold=scoring_threshold)
    return waves, (coefs, genes, scaletime)


def pipeline2(data_loader: callable,
              window = None,
             window_params: dict = None,
             trajectory: list = None,
             scoring_threshold: float = 1,
              exclude_pt_ends: tuple = (0.05, 0.95),
              repeat: bool = False
             ) -> sc.AnnData:
    if os.path.exists("psuper_model.h5ad") and not repeat:
        adata = sc.read_h5ad("psuper_model.h5ad")
    else:
        adata = data_loader()
        print(adata)

        print("Preprocessing...")
        sc.pp.filter_cells(adata, min_genes=2000, inplace=True)
        sc.pp.filter_genes(adata, min_cells=3, inplace=True)

        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        sc.pp.regress_out(adata, ['n_genes'])
        sc.pp.scale(adata)
        sc.pp.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.leiden(adata, resolution=1)

        print("Computing pseudotime...")
        cell_cycle_genes = [x.strip() for x in open('regev_lab_cell_cycle_genes.txt')]
        s_genes = cell_cycle_genes[:43]
        g2m_genes = cell_cycle_genes[43:]
        cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
        phase_order = {'G1': 1, 'S': 2, 'G2M': 3}
        adata.obs['phase_ordinal'] = adata.obs['phase'].map(phase_order)

        psuper = Psupertime()
        # Fit the model
        psuper.run(adata, 'phase_ordinal')
        # psuper.plot_labels_over_psupertime(adata, "phase_ordinal")
        psuper.predict_psuper(adata)
        if not os.path.exists("psuper_model.h5ad"):
            adata.write_h5ad("psuper_model.h5ad")
    print("Computing gene expression along pseudotime...")
    # Do windowing
    if window is None:
        if window_params is None:
            window_params = {"n_windows": 30, "sigma": 0.03, "max_distance": 0.1}
        window = wavelet_pseudotime.windowing.ConfinedGaussianWindow(**window_params)
    n_windows = window.n_windows
    qmin = adata.obs['psupertime'].quantile(exclude_pt_ends[0])
    qmax = adata.obs['psupertime'].quantile(exclude_pt_ends[1])
    adata = adata[(adata.obs["psupertime"] > qmin) & (adata.obs["psupertime"] < qmax), :].copy()
    if trajectory is None:  # take all
        node_col = "phase"
        path = list(np.unique(adata.obs[node_col]))
    pseudotime_signals_array = window_trajectory(adata,
                                                  trajectory=path,
                                                  node_col=node_col,
                                                  window=window,
                                                  pseudotime_col='psupertime',
                                                  exclude_minimum=False)
    pseudotime_signals_dict = {}
    for idx, g in enumerate(adata.var_names):
        pseudotime_signals_dict[g] = pseudotime_signals_array[:, idx]
    wt = WaveletTransform(scales=np.arange(1, 7), wavelet="mexh")
    waves, best_tuple = get_max_scored_wavelets(pseudotime_signals_dict,
                                                wt,
                                                std_from_median,
                                                score_threshold=scoring_threshold,
                                                top_fraction=0.01,
                                                )
    return waves, best_tuple, pseudotime_signals_dict, adata


def pipeline3(adata,
              window = None,
             window_params: dict = None,
             trajectory: list = None,
             scoring_threshold: float = 1,
              exclude_pt_ends: tuple = (0.05, 0.95),
              repeat: bool = False
             ) -> sc.AnnData:
    # if os.path.exists("psuper_model.h5ad") and not repeat:
    #     adata = sc.read_h5ad("psuper_model.h5ad")
    # else:
    #     adata = data_loader()
    #     print(adata)
    #
    #     print("Preprocessing...")
    #     sc.pp.filter_cells(adata, min_genes=2000, inplace=True)
    #     sc.pp.filter_genes(adata, min_cells=3, inplace=True)
    #
    #     sc.pp.normalize_total(adata, inplace=True)
    #     sc.pp.log1p(adata)
    #     sc.pp.regress_out(adata, ['n_genes'])
    #     sc.pp.scale(adata)
    #     sc.pp.pca(adata, svd_solver="arpack")
    #     sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    #     sc.tl.leiden(adata, resolution=1)
    #
    #     print("Computing pseudotime...")
    #     cell_cycle_genes = [x.strip() for x in open('regev_lab_cell_cycle_genes.txt')]
    #     s_genes = cell_cycle_genes[:43]
    #     g2m_genes = cell_cycle_genes[43:]
    #     cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
    #     sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    #     phase_order = {'G1': 1, 'S': 2, 'G2M': 3}
    #     adata.obs['phase_ordinal'] = adata.obs['phase'].map(phase_order)
    #
    #     psuper = Psupertime()
    #     # Fit the model
    #     psuper.run(adata, 'phase_ordinal')
    #     # psuper.plot_labels_over_psupertime(adata, "phase_ordinal")
    #     psuper.predict_psuper(adata)
    #     if not os.path.exists("psuper_model.h5ad"):
    #         adata.write_h5ad("psuper_model.h5ad")
    print("Computing gene expression along pseudotime...")
    # Do windowing
    if window is None:
        if window_params is None:
            window_params = {"n_windows": 30, "sigma": 0.03, "max_distance": 0.1}
        window = wavelet_pseudotime.windowing.ConfinedGaussianWindow(**window_params)
    n_windows = window.n_windows
    qmin = adata.obs['psupertime'].quantile(exclude_pt_ends[0])
    qmax = adata.obs['psupertime'].quantile(exclude_pt_ends[1])
    adata = adata[(adata.obs["psupertime"] > qmin) & (adata.obs["psupertime"] < qmax), :].copy()
    if trajectory is None:  # take all
        node_col = "phase"
        path = list(np.unique(adata.obs[node_col]))
    pseudotime_signals_array = window_trajectory(adata,
                                                  trajectory=path,
                                                  node_col=node_col,
                                                  window=window,
                                                  pseudotime_col='psupertime',
                                                  exclude_minimum=False)
    pseudotime_signals_dict = {}
    for idx, g in enumerate(adata.var_names):
        pseudotime_signals_dict[g] = pseudotime_signals_array[:, idx]
    wt = WaveletTransform(scales=np.arange(1, 7), wavelet="mexh")
    waves, scores = get_scored_wavelet_across_sigs(pseudotime_signals_dict,
                                                   wt,
                                                   mag_median,
                                                   score_threshold=scoring_threshold,
                                                   normalize_signal=False)
    return waves, scores, pseudotime_signals_dict, adata
    # waves, best_tuple = get_max_scored_wavelets(pseudotime_signals_dict,
    #                                             wt,
    #                                             std_from_median,
    #                                             score_threshold=scoring_threshold,
    #                                             top_fraction=0.01,
    #                                             normalize_signal=False
    #                                             )
    # return waves, best_tuple, pseudotime_signals_dict, adata


def pipeline4(data_loader,
              window = None,
             window_params: dict = None,
             trajectory: list = None,
             scoring_threshold: float = 1,
              exclude_pt_ends: tuple = (0.05, 0.95),
              repeat: bool = False,
              save_name: str = None,
              coverage_threshold: float = 0.0,
             ) -> sc.AnnData:
    if os.path.exists(save_name) and not repeat:
        adata = sc.read_h5ad(save_name)
    else:
        adata = data_loader()
        print(adata)

        print("Preprocessing...")
        sc.pp.filter_cells(adata, min_genes=2000, inplace=True)
        sc.pp.filter_genes(adata, min_cells=3, inplace=True)
        gene_coverage(adata)
        adata = adata[:, adata.var["coverage"] > coverage_threshold].copy()
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        sc.pp.regress_out(adata, ['n_genes'])
        sc.pp.scale(adata)
        sc.pp.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.leiden(adata, resolution=1)

        print("Computing pseudotime...")
        cell_cycle_genes = [x.strip() for x in open('regev_lab_cell_cycle_genes.txt')]
        s_genes = cell_cycle_genes[:43]
        g2m_genes = cell_cycle_genes[43:]
        cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
        phase_order = {'G1': 1, 'S': 2, 'G2M': 3}
        adata.obs['phase_ordinal'] = adata.obs['phase'].map(phase_order)

        psuper = Psupertime()
        # Fit the model
        psuper.run(adata, 'phase_ordinal')
        # psuper.plot_labels_over_psupertime(adata, "phase_ordinal")
        psuper.predict_psuper(adata)
        adata.write_h5ad(save_name)
    print("Computing gene expression along pseudotime...")
    # Do windowing
    if window is None:
        if window_params is None:
            window_params = {"n_windows": 30, "sigma": 0.03, "max_distance": 0.1}
        window = wavelet_pseudotime.windowing.ConfinedGaussianWindow(**window_params)
    n_windows = window.n_windows
    qmin = adata.obs['psupertime'].quantile(exclude_pt_ends[0])
    qmax = adata.obs['psupertime'].quantile(exclude_pt_ends[1])
    adata = adata[(adata.obs["psupertime"] > qmin) & (adata.obs["psupertime"] < qmax), :].copy()
    if trajectory is None:  # take all
        node_col = "phase"
        path = list(np.unique(adata.obs[node_col]))
    pseudotime_signals_array = window_trajectory(adata,
                                                  trajectory=path,
                                                  node_col=node_col,
                                                  window=window,
                                                  pseudotime_col='psupertime',
                                                  exclude_minimum=False)
    pseudotime_signals_dict = {}
    for idx, g in enumerate(adata.var_names):
        pseudotime_signals_dict[g] = pseudotime_signals_array[:, idx]
    wt = WaveletTransform(scales=np.arange(1, 7), wavelet="mexh")
    waves, scores = get_scored_wavelet_across_sigs(pseudotime_signals_dict,
                                                   wt,
                                                   mag_median,
                                                   score_threshold=scoring_threshold,
                                                   normalize_signal=False)
    return waves, scores, pseudotime_signals_dict, adata


def pipeline5(data_loader,
              window = None,
             window_params: dict = None,
             trajectory: list = None,
             scoring_threshold: float = 1,
              exclude_pt_ends: tuple = (0.05, 0.95),
              repeat: bool = False,
              save_name: str = None
             ) -> sc.AnnData:
    if os.path.exists(save_name) and not repeat:
        adata = sc.read_h5ad(save_name)
    else:
        adata = data_loader()
        print(adata)

        print("Preprocessing...")
        sc.pp.filter_cells(adata, min_genes=2000, inplace=True)
        sc.pp.filter_genes(adata, min_cells=3, inplace=True)

        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        sc.pp.regress_out(adata, ['n_genes'])
        sc.pp.scale(adata)
        sc.pp.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.leiden(adata, resolution=1)

        print("Computing pseudotime...")
        cell_cycle_genes = [x.strip() for x in open('regev_lab_cell_cycle_genes.txt')]
        s_genes = cell_cycle_genes[:43]
        g2m_genes = cell_cycle_genes[43:]
        cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
        phase_order = {'G1': 1, 'S': 2, 'G2M': 3}
        adata.obs['phase_ordinal'] = adata.obs['phase'].map(phase_order)

        psuper = Psupertime()
        # Fit the model
        psuper.run(adata, 'phase_ordinal')
        # psuper.plot_labels_over_psupertime(adata, "phase_ordinal")
        psuper.predict_psuper(adata)
        adata.write_h5ad(save_name)
    print("Computing gene expression along pseudotime...")
    # Do windowing
    if window is None:
        if window_params is None:
            window_params = {"n_windows": 30, "sigma": 0.03, "max_distance": 0.1}
        window = wavelet_pseudotime.windowing.ConfinedGaussianWindow(**window_params)
    n_windows = window.n_windows
    qmin = adata.obs['psupertime'].quantile(exclude_pt_ends[0])
    qmax = adata.obs['psupertime'].quantile(exclude_pt_ends[1])
    adata = adata[(adata.obs["psupertime"] > qmin) & (adata.obs["psupertime"] < qmax), :].copy()
    if trajectory is None:  # take all
        node_col = "phase"
        path = list(np.unique(adata.obs[node_col]))
    pseudotime_signals_array = window_trajectory(adata,
                                                  trajectory=path,
                                                  node_col=node_col,
                                                  window=window,
                                                  pseudotime_col='psupertime',
                                                  exclude_minimum=False)
    pseudotime_signals_dict = {}
    for idx, g in enumerate(adata.var_names):
        pseudotime_signals_dict[g] = pseudotime_signals_array[:, idx]
    wt = WaveletTransform(scales=np.arange(1, 7), wavelet="mexh")
    waves, scores = get_scored_wavelet_across_sigs(pseudotime_signals_dict,
                                                   wt,
                                                   mag_median,
                                                   score_threshold=scoring_threshold,
                                                   normalize_signal=False)
    return waves, scores, pseudotime_signals_dict, adata


def pipeline_paul15v2(adata,
              window = None,
             window_params: dict = None,
             trajectory: list = None,
             scoring_threshold: float = 1,
              exclude_pt_ends: tuple = (0.05, 0.95),
                      node_col: str = "paul15_clusters",
              repeat: bool = False
             ) -> sc.AnnData:
    # if os.path.exists("psuper_model.h5ad") and not repeat:
    #     adata = sc.read_h5ad("psuper_model.h5ad")
    # else:
    #     adata = data_loader()
    #     print(adata)
    #
    #     print("Preprocessing...")
    #     sc.pp.filter_cells(adata, min_genes=2000, inplace=True)
    #     sc.pp.filter_genes(adata, min_cells=3, inplace=True)
    #
    #     sc.pp.normalize_total(adata, inplace=True)
    #     sc.pp.log1p(adata)
    #     sc.pp.regress_out(adata, ['n_genes'])
    #     sc.pp.scale(adata)
    #     sc.pp.pca(adata, svd_solver="arpack")
    #     sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    #     sc.tl.leiden(adata, resolution=1)
    #
    #     print("Computing pseudotime...")
    #     cell_cycle_genes = [x.strip() for x in open('regev_lab_cell_cycle_genes.txt')]
    #     s_genes = cell_cycle_genes[:43]
    #     g2m_genes = cell_cycle_genes[43:]
    #     cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
    #     sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    #     phase_order = {'G1': 1, 'S': 2, 'G2M': 3}
    #     adata.obs['phase_ordinal'] = adata.obs['phase'].map(phase_order)
    #
    #     psuper = Psupertime()
    #     # Fit the model
    #     psuper.run(adata, 'phase_ordinal')
    #     # psuper.plot_labels_over_psupertime(adata, "phase_ordinal")
    #     psuper.predict_psuper(adata)
    #     if not os.path.exists("psuper_model.h5ad"):
    #         adata.write_h5ad("psuper_model.h5ad")
    print("Computing gene expression along pseudotime...")
    # Do windowing
    if window is None:
        if window_params is None:
            window_params = {"n_windows": 30, "sigma": 0.03, "max_distance": 0.1}
        window = wavelet_pseudotime.windowing.ConfinedGaussianWindow(**window_params)
    n_windows = window.n_windows
    qmin = adata.obs['psupertime'].quantile(exclude_pt_ends[0])
    qmax = adata.obs['psupertime'].quantile(exclude_pt_ends[1])
    print(f"Pre exclusion: {adata.shape}")
    adata = adata[(adata.obs["psupertime"] > qmin) & (adata.obs["psupertime"] < qmax), :].copy()
    print(f"Post exclusion: {adata.shape}")
    if trajectory is None:  # take all
        node_col = "phase"
        path = list(np.unique(adata.obs[node_col]))
    pseudotime_signals_array = window_trajectory(adata,
                                                  trajectory=trajectory,
                                                  node_col=node_col,
                                                  window=window,
                                                  pseudotime_col='psupertime',
                                                  exclude_minimum=False)
    pseudotime_signals_dict = {}
    for idx, g in enumerate(adata.var_names):
        pseudotime_signals_dict[g] = pseudotime_signals_array[:, idx]
    wt = WaveletTransform(scales=np.arange(1, 7), wavelet="mexh")
    waves, scores = get_scored_wavelet_across_sigs(pseudotime_signals_dict,
                                                   wt,
                                                   mag_median,
                                                   score_threshold=scoring_threshold,
                                                   normalize_signal=False)
    return waves, scores, pseudotime_signals_dict, adata


def pipeline_astral_cellcycle(data_loader,
              window = None,
             window_params: dict = None,
             trajectory: list = None,
             scoring_threshold: float = 1,
              exclude_pt_ends: tuple = (0.05, 0.95),
              repeat: bool = False,
              save_name: str = None,
              coverage_threshold: float = 0.0,
             ) -> sc.AnnData:
    print("5")
    if os.path.exists(save_name) and not repeat:
        adata = sc.read_h5ad(save_name)
    else:
        adata = data_loader()
        print("Preprocessing...")
        sc.pp.filter_cells(adata, min_genes=2000, inplace=True)
        sc.pp.filter_genes(adata, min_cells=3, inplace=True)
        # gene_coverage(adata)
        # adata = adata[:, adata.var["coverage"] > coverage_threshold].copy()
        # sc.pp.normalize_total(adata, inplace=True)
        sc.pp.normalize_total(adata, inplace=True, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.regress_out(adata, ['n_genes'])
        # sc.pp.scale(adata)
        sc.pp.scale(adata, max_value=10)
        sc.pp.pca(adata, svd_solver="arpack")
        # sc.tl.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.leiden(adata, resolution=1)

        print("Computing pseudotime...")
        f = open("regev_lab_cell_cycle_genes.txt", "r")
        cell_cycle_genes = [x.strip() for x in f]
        f.close()
        s_genes = cell_cycle_genes[:43]
        g2m_genes = cell_cycle_genes[43:]
        cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
        phase_order = {'G1': 1, 'S': 2, 'G2M': 3}
        # phase_order = {"G1": 1, "S": 3, "G2M": 2}
        adata.obs['phase_ordinal'] = adata.obs['phase'].map(phase_order)

        psuper = Psupertime()
        # Fit the model
        psuper.run(adata, 'phase_ordinal')
        # psuper.plot_labels_over_psupertime(adata, "phase_ordinal")
        psuper.predict_psuper(adata)
        adata.write_h5ad(save_name)
    print("Computing gene expression along pseudotime...")
    # Do windowing
    if window is None:
        if window_params is None:
            window_params = {"n_windows": 30, "sigma": 0.03, "max_distance": 0.1}
        window = wavelet_pseudotime.windowing.ConfinedGaussianWindow(**window_params)
    n_windows = window.n_windows
    qmin = adata.obs['psupertime'].quantile(exclude_pt_ends[0])
    qmax = adata.obs['psupertime'].quantile(exclude_pt_ends[1])
    adata = adata[(adata.obs["psupertime"] > qmin) & (adata.obs["psupertime"] < qmax), :].copy()
    if trajectory is None:  # take all
        node_col = "phase"
        path = list(np.unique(adata.obs[node_col]))
    pseudotime_signals_array = window_trajectory(adata,
                                                  trajectory=path,
                                                  node_col=node_col,
                                                  window=window,
                                                  pseudotime_col='psupertime',
                                                  exclude_minimum=False)
    pseudotime_signals_dict = {}
    for idx, g in enumerate(adata.var_names):
        pseudotime_signals_dict[g] = pseudotime_signals_array[:, idx]
    wt = WaveletTransform(scales=np.arange(1, 5), wavelet="mexh")
    waves, scores = get_scored_wavelet_across_sigs(pseudotime_signals_dict,
                                                   wt,
                                                   mag_median,
                                                   score_threshold=scoring_threshold,
                                                   normalize_signal=False)
    return waves, scores, pseudotime_signals_dict, adata, psuper


def pipeline_astral_cellcycle2(data_loader,
              window = None,
             window_params: dict = None,
             trajectory: list = None,
             scoring_threshold: float = 1,
              exclude_pt_ends: tuple = (0.05, 0.95),
              repeat: bool = False,
              save_name: str = None,
              coverage_threshold: float = 0.0,
             ) -> sc.AnnData:
    print("4")
    if os.path.exists(save_name) and not repeat:
        adata = sc.read_h5ad(save_name)
    else:
        adata = data_loader()
        print(adata)

        print(adata.shape)
        # Filter out cells with very few detected proteins
        sc.pp.filter_cells(adata, min_genes=2000, inplace=True)  # Adjust threshold as needed

        # Filter out proteins detected in very few cells
        sc.pp.filter_genes(adata, min_cells=3)  # Adjust threshold as needed
        gene_coverage(adata)
        adata = adata[:, adata.var["coverage"] > coverage_threshold].copy()
        print(adata.shape)
        # Normalize total protein counts per cell to 10,000 (or another scale)
        sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)

        # Log transformation to reduce variability
        sc.pp.log1p(adata)
        sc.pp.regress_out(adata, ['n_genes'])  # Remove variability from total protein counts
        sc.pp.scale(adata, max_value=10)  # Cap extreme values
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)  # Adjust n_neighbors as needed
        sc.tl.leiden(adata, resolution=0.6, random_state=42)
        sc.tl.umap(adata)
        cell_cycle_genes = [x.strip() for x in open('regev_lab_cell_cycle_genes.txt')]
        s_genes = cell_cycle_genes[:43]
        g2m_genes = cell_cycle_genes[43:]
        cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
        # Make the UMAP plot
        # sc.pl.umap(adata,
        #            color=['leiden', 'n_genes', 'phase'])  # Color by total protein counts or another metadata field
        # # Make the UMAP plot
        # sc.pl.umap(adata,
        #            color=['leiden', 'n_genes', 'phase'])  # Color by total protein counts or another metadata field
        phase_order = {'G1': 1, 'S': 2, 'G2M': 3}

        # Map the phase column to ordinal values
        adata.obs['phase_ordinal'] = adata.obs['phase'].map(phase_order)

        # Ensure 'phase' is a categorical variable with the correct order
        # adata.obs['phase'] = adata.obs['phase'].astype('category')

        # Initialize Psupertime with the AnnData object and the 'phase' labels
        psuper = Psupertime()

        # Fit the model
        psuper.run(adata, 'phase_ordinal')
        psuper.predict_psuper(adata)

    print("Computing gene expression along pseudotime...")
    # Do windowing
    if window is None:
        if window_params is None:
            window_params = {"n_windows": 30, "sigma": 0.03, "max_distance": 0.1}
        window = wavelet_pseudotime.windowing.ConfinedGaussianWindow(**window_params)
    n_windows = window.n_windows
    qmin = adata.obs['psupertime'].quantile(exclude_pt_ends[0])
    qmax = adata.obs['psupertime'].quantile(exclude_pt_ends[1])
    adata = adata[(adata.obs["psupertime"] > qmin) & (adata.obs["psupertime"] < qmax), :].copy()
    if trajectory is None:  # take all
        node_col = "phase"
        path = list(np.unique(adata.obs[node_col]))
    pseudotime_signals_array = window_trajectory(adata,
                                                  trajectory=path,
                                                  node_col=node_col,
                                                  window=window,
                                                  pseudotime_col='psupertime',
                                                  exclude_minimum=False)
    pseudotime_signals_dict = {}
    for idx, g in enumerate(adata.var_names):
        pseudotime_signals_dict[g] = pseudotime_signals_array[:, idx]
    wt = WaveletTransform(scales=np.arange(1, 5), wavelet="mexh")
    waves, scores = get_scored_wavelet_across_sigs(pseudotime_signals_dict,
                                                   wt,
                                                   mag_median,
                                                   score_threshold=scoring_threshold,
                                                   normalize_signal=False)
    return waves, scores, pseudotime_signals_dict, adata, psuper


def pipeline_slavov(adata,
              window = None,
             window_params: dict = None,
             trajectory: list = None,
             scoring_threshold: float = 1,
              exclude_pt_ends: tuple = (0.05, 0.95),
              repeat: bool = False,
              save_name: str = None,
              coverage_threshold: float = 0.0,
             ) -> sc.AnnData:
    if os.path.exists(save_name) and not repeat:
        adata = sc.read_h5ad(save_name)
    else:
        # data_loader()
        print(adata)

        print("Preprocessing...")
        print(adata.shape)
        # sc.pp.filter_cells(adata, min_genes=100, inplace=True)
        # sc.pp.filter_genes(adata, min_cells=3, inplace=True)
        # # gene_coverage(adata)
        # # adata = adata[:, adata.var["coverage"] > coverage_threshold].copy()
        # # sc.pp.normalize_total(adata, inplace=True)
        # print(adata.shape)
        # sc.pp.normalize_total(adata, inplace=True, target_sum=1e4)
        # sc.pp.log1p(adata)
        # sc.pp.regress_out(adata, ['n_genes'])
        # # sc.pp.scale(adata)
        # sc.pp.scale(adata, max_value=10)
        sc.pp.pca(adata, svd_solver="arpack")
        # sc.tl.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=20, n_pcs=40)
        sc.tl.leiden(adata, resolution=1)

        print("Computing pseudotime...")
        sc.tl.dpt(adata)
        # f = open("regev_lab_cell_cycle_genes.txt", "r")
        # cell_cycle_genes = [x.strip() for x in f]
        # f.close()
        # s_genes = cell_cycle_genes[:43]
        # g2m_genes = cell_cycle_genes[43:]
        # cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
        # sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
        # phase_order = {'G1': 1, 'S': 2, 'G2M': 3}
        # adata.obs['phase_ordinal'] = adata.obs['phase'].map(phase_order)

        # psuper = Psupertime()
        # # Fit the model
        # psuper.run(adata, 'phase_ordinal')
        # # psuper.plot_labels_over_psupertime(adata, "phase_ordinal")
        # psuper.predict_psuper(adata)
        adata.write_h5ad(save_name)
    print("Computing gene expression along pseudotime...")
    # Do windowing
    if window is None:
        if window_params is None:
            window_params = {"n_windows": 30, "sigma": 0.03, "max_distance": 0.1}
        window = wavelet_pseudotime.windowing.ConfinedGaussianWindow(**window_params)
    n_windows = window.n_windows
    adata.obs["psupertime"] = adata.obs["dpt_pseudotime"]
    qmin = adata.obs['dpt_pseudotime'].quantile(exclude_pt_ends[0])
    qmax = adata.obs['dpt_pseudotime'].quantile(exclude_pt_ends[1])
    print(f"qmin: {qmin}")
    print(f"qmax: {qmax}")
    adata = adata[(adata.obs["dpt_pseudotime"] > qmin) & (adata.obs["dpt_pseudotime"] < qmax), :].copy()
    sc.tl.leiden(adata, resolution=1)
    if trajectory is None:  # take all
        node_col = "leiden"
        path = list(np.unique(adata.obs[node_col]))
    pseudotime_signals_array = window_trajectory(adata,
                                                  trajectory=path,
                                                  node_col=node_col,
                                                  window=window,
                                                  pseudotime_col='dpt_pseudotime',
                                                  exclude_minimum=False)
    pseudotime_signals_dict = {}
    for idx, g in enumerate(adata.var_names):
        pseudotime_signals_dict[g] = pseudotime_signals_array[:, idx]
    wt = WaveletTransform(scales=np.arange(1, 5), wavelet="mexh")
    waves, scores = get_scored_wavelet_across_sigs(pseudotime_signals_dict,
                                                   wt,
                                                   mag_median,
                                                   score_threshold=scoring_threshold,
                                                   normalize_signal=False)
    return waves, scores, pseudotime_signals_dict, adata