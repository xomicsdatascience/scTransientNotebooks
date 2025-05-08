from matplotlib import pyplot as plt
import anndata as ad
from anndata import AnnData
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.sparse import csr_matrix
from typing import Optional


def draw_path(coordinates: np.array,
              path: list,
              connectivity_matrix: Optional[csr_matrix] = None,
              node_labels: Optional[list] = None,
              ax = None):
    """
    This function receives a list of coordinates and a list of nodes representing from a path from start to end

    Parameters
    ----------
    coordinates : np.array
      Array with (x, y) coordinates
    path : list
        List of int with indices representing the path
    connectivity_matrix : csr_matrix, Optional
        Connectivity matrix used for line width.
    node_labels : list, Optional
        Labels associated with nodes; assumes that the ordering matches the order of the coordinates.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if connectivity_matrix is None:
        connectivity_matrix = np.ones((coordinates.shape[0], coordinates.shape[0]))
    for i in range(coordinates.shape[0]):
        x, y = coordinates[i, :]
        ax.plot(x, y, "o")
        if node_labels is None:
            ax.text(x, y, str(i))
        else:
            ax.text(x, y, str(node_labels[i]))
    for i in range(len(path) - 1):
        start = coordinates[path[i], :]
        end = coordinates[path[i + 1], :]
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=1+3*connectivity_matrix[path[i], path[i + 1]])
    return ax


def show_pt_distribution(df: pd.DataFrame,
                         num_bins: int = 30,
                         group_values: list = None,
                         pseudotime_label: str = "dpt_pseudotime",
                         group_label: str = None):
    """
    Shows the distribution of pseudotime for each label in the group.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing pseudotime values for samples and the group labels of interest.
    num_bins : int
        Number of bins to use for the histogram.
    group_values : list
        Limit the plotted distributions to the specified values.
    pseudotime_label : str
        Label for the pseudotime values in `df`
    group_label : str
        Label for the group of interest. If None and df.shape[1] == 2, then the non-pseudotime_label column
        is taken as the group label.
    """
    if group_label is None:
        if df.shape[1] != 2:
            raise ValueError("Group label must be specified for DataFrames containing more than 2 columns.")
        labels = df.columns
        group_label = labels[0] if labels[0] != pseudotime_label else labels[1]
    if group_values is None:

        group_values = list(df[group_label].cat.categories)
    hist_dict = dict()
    mean_dict = dict()
    # drop inf values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[pseudotime_label])
    bins = np.linspace(0,1, num_bins)
    for gv in group_values:
        h, _ = np.histogram(df[df[group_label] == gv][pseudotime_label], bins=bins)
        hist_dict[gv] = h
        mean_dict[gv] = np.mean(df[df[group_label] == gv][pseudotime_label])
    for k, v in hist_dict.items():
        plt.plot(np.linspace(0,1,len(v)),v, label=k)
    plt.legend()
    return


def heatmap_timecourses(adata: AnnData,
                        path: list,
                        gene_list: list,
                        save: str = None):
    """
    Recreates sc.pl.paga_path because for some reason it gives key errors despite the genes being in var_names
    Parameters
    ----------
    adata : AnnData
        AnnData object.
    path : list
        List of leiden cluster values along which to plot the gene expression.
    gene_list : list
        List of genes to plot.
    save : Optional[str]
        If not None, saves the figure at the filepath.
    """
    dpt_ordered_idx = list(adata.obs.sort_values(by=["dpt_pseudotime"], ascending=True).index)
    adata = adata[dpt_ordered_idx, :]
    ordered_idx = []
    cluster_extents = []
    leiden_values = []
    for p in path:
        leiden_subset = adata.obs.loc[adata.obs["leiden"] == p]
        leiden_values.extend(list(leiden_subset["dpt_pseudotime"]))
        ordered_idx.extend(list(leiden_subset.index))
        cluster_extents.append(len(leiden_subset))

    cluster_extents = np.cumsum([c/len(ordered_idx) for c in cluster_extents])
    plt.figure()
    shapes = adata[ordered_idx, gene_list].shape
    new_data = np.zeros((shapes[0], shapes[1]+1))
    data = adata[ordered_idx, gene_list].X
    for i in range(data.shape[1]):
        new_data[:, i] = (data[:, i ] - np.mean(data[:,i])) / np.std(data[:,i])
    mmin = np.min(new_data[:,:-1])
    mmax = np.max(new_data[:,:-1])
    new_data[:, -1] = mmin + mmax*(np.array(leiden_values) - min(leiden_values)) / (max(leiden_values)-min(leiden_values))

    heatmap = sns.heatmap(new_data.T, cmap=sns.color_palette("viridis", as_cmap=True))
    heatmap.set_yticklabels(gene_list + ["pseudotime"], rotation=90)
    heatmap.set_xticks([])
    plt.yticks(rotation=0)
    _add_rectangles(heatmap, path, cluster_extents)
    if save is not None:
        plt.savefig(save)
    plt.show()
    return


def _add_rectangles(heatmap,
                    path,
                    cluster_extents):
    """Adds rectangles showing the extent of the clusters."""
    bottom = -1
    xlims = heatmap.get_xlim()
    x_extents = [0] + [(xlims[1]-xlims[0])*c + xlims[0] for c in cluster_extents]
    cmap = plt.get_cmap('tab10')
    for idx, p in enumerate(path):
        heatmap.add_patch(Rectangle((x_extents[idx], bottom), x_extents[idx+1] - x_extents[idx], 1, edgecolor='k', facecolor=cmap(idx)))
        heatmap.annotate(p,
                         ((x_extents[idx+1] + x_extents[idx])/2, bottom/2),
                         weight="bold")
    ylims = heatmap.get_ylim()
    heatmap.set_ylim((ylims[0], bottom))
    return

