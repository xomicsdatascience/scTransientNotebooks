import numpy as np
import pandas as pd
import scanpy as sc


def quantile_binning(x, y, num_bins=50,
                     exclude_minimum: bool = False):
    if isinstance(x, pd.Series):
        x = x.values
    data = pd.DataFrame(y, dtype=np.float32)
    data["_xbins"], bins = pd.qcut(x, q=num_bins, duplicates="drop", retbins=True)
    if exclude_minimum:
        binned = data.groupby("_xbins", observed=True).apply(
            lambda df: df.drop(columns="_xbins").apply(
                lambda col: col[col != col.min()].mean()
            )
        ).reset_index()
        return binned, bins
    return data.groupby("_xbins", observed=True).mean().reset_index(), bins


def quantile_binning_anndata(adata,
                             pseudotime_key: str = "dpt_pseudotime",
                             num_bins=50,
                             exclude_minimum: bool = False):
    binned, bins = quantile_binning(adata.obs[pseudotime_key].values,
                            adata.X,
                            num_bins=num_bins,
                                    exclude_minimum=exclude_minimum)
    remap = {}
    for idx, v in enumerate(adata.var_names):
        remap[idx] = v
    binned.rename(columns=remap, inplace=True)
    return binned, bins