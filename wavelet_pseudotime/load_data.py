import scanpy as sc
import numpy as np
import pandas as pd
import mygene
import os
from anndata import AnnData
from os.path import join
from typing import List, Literal
import anndata as ad
data_dir = os.path.dirname(__file__) + "/../data"


def load_paul15(process_data: bool = True) -> sc.AnnData:
    """
    Loads and preprocesses the Paul15 dataset.
    Parameters
    ----------
    process_data : bool
        Whether to preprocess the data.

    Returns
    -------
    AnnData
        Paul15 dataset.
    """
    adata = sc.datasets.paul15()
    if process_data:
        gene_coverage(adata)
        retain_genes = adata.var["coverage"] > 0.5
        adata = adata[:, retain_genes]
        sc.pp.normalize_total(adata)  # Normalize to some total.
        sc.pp.log1p(adata)  # Take the natural log of the data.
        sc.pp.scale(adata)  # rescale to 0 mean and variance of 1
        sc.pp.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
        sc.tl.leiden(adata, resolution=1)
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata)
        sc.tl.paga(adata)
    return adata


def gene_coverage(adata: AnnData,
                  min_as_baseline: bool = True):
    """
    Computes the gene coverage of each gene
    Parameters
    ----------
    adata
    min_as_baseline : bool
        Whether to use the minimum gene expression as the baseline.

    Returns
    -------

    """
    if min_as_baseline:
        baseline = adata.X.min(axis=0)
    else:
        baseline = 0
    adata.var["coverage"] = np.sum(adata.X > baseline, axis=0) / adata.X.shape[0]
    return


def load_triana(process_data: bool = True) -> sc.AnnData:
    """
    Load and preprocess the Triana dataset.
    Parameters
    ----------
    process_data : bool
        Whether to preprocess the data.

    Returns
    -------
    AnnData
        Triana dataset.
    """
    adata = sc.read_h5ad(f"{data_dir}/triana_3healthy.h5ad")

    if process_data:
        remap_ensembl(adata)
        hsc0 = np.where(adata.obs["cell_type"] == "hematopoietic multipotent progenitor cell")[0][0]
        # eryth0 = np.where(adata_triana.obs["cell_type"] == "erythroid lineage cell")[0][0]
        # mono0 = np.where(adata_triana.obs["cell_type"] == "classical monocyte")[0][0]
        adata.uns["iroot"] = hsc0

        sc.pp.log1p(adata)
        sc.pp.scale(adata)  # rescale to 0 mean and variance of 1
        sc.pp.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=20, n_pcs=20)
        sc.tl.diffmap(adata)  # compute diffusion map
        sc.tl.dpt(adata)  # compute diffusion pseudotime
        sc.tl.leiden(adata, resolution=1)
        sc.tl.paga(adata)
    return adata


def load_embryo2018(process_data: bool = True) -> sc.AnnData:
    """
    Load the Embryo2018 dataset from the Harmony paper.
    Parameters
    ----------
    process_data : bool
        Whether to preprocess the data.

    Returns
    -------

    """
    adata = sc.read_mtx(f"{data_dir}/embryo2018/atlas/raw_counts.mtx")
    genes = pd.read_csv(f"{data_dir}/embryo2018/atlas/genes.tsv", header=None, sep="\t")[1].values
    adata = adata.T
    adata.var_names = [g.lower() for g in genes]

    obs = pd.read_csv(f"{data_dir}/embryo2018/atlas/meta.tab", sep="\t")
    adata.obs = obs[["celltype"]]

    if process_data:
        adata.uns["iroot"] = 0
        sc.pp.normalize_total(adata)  # Normalize to some total.
        sc.pp.log1p(adata)  # Take the natural log of the data.
        sc.pp.scale(adata)  # rescale to 0 mean and variance of 1
        sc.pp.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=1)
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata)
        sc.tl.paga(adata)

    return adata


def load_slavov(process_data: bool = True) -> sc.AnnData:
    """Load the Slavov Scope2 data for macrophages"""
    cell_path = "data/slavov/Cells.csv"
    protein_path = "data/slavov/Proteins-processed.csv"
    protein_mapping = "data/slavov/protein_mapping.tsv"
    adata_obs = pd.read_csv(cell_path, header=None).T
    adata_obs.columns = adata_obs.iloc[0]
    adata_obs.set_index(adata_obs.iloc[:, 0], inplace=True)
    adata_obs.drop(np.nan, inplace=True)
    adata_obs.drop(np.nan, axis=1, inplace=True)

    adata_data = pd.read_csv(protein_path, low_memory=False, header=None).T
    adata_data.columns = adata_data.iloc[0]
    adata_data.set_index(adata_data.iloc[:, 0], inplace=True)
    adata_data.drop(np.nan, inplace=True)
    adata_data.drop('protein', inplace=True)
    adata_data.drop(np.nan, axis=1, inplace=True)

    var_name_map = pd.read_csv(protein_mapping, header=None, sep="\t", index_col=0).to_dict()[1]
    # remap column names from adata_data using var_name_map
    adata_data.rename(columns=var_name_map, inplace=True)

    adata_data = adata_data.astype(float)
    adata = sc.AnnData(adata_data, obs=adata_obs)

    if not process_data:
        return adata
    sc.pp.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50)
    sc.tl.diffmap(adata)  # compute diffusion map
    # sc.tl.dpt(adata)  # compute diffusion pseudotime
    sc.tl.leiden(adata, resolution=0.5)
    sc.tl.paga(adata)
    return adata


def load_slavov_xomic() -> (sc.AnnData, sc.AnnData):
    path = "data/slavov/joint.csv"
    data = pd.read_csv(path, index_col=0, header=0).T
    id_mapping = pd.read_csv("data/slavov/idmapping_2025_03_14.tsv", sep="\t")
    id_dict = id_mapping.set_index('From')['To'].to_dict()

    prot_rows = []
    tscr_rows = []
    print(data.shape)
    for c in list(data.index):
        if "i" in c:
            prot_rows.append(c)
        if "A" in c or "C" in c or "T" in c or "G" in c:
            tscr_rows.append(c)

    data_tscr = data.loc[tscr_rows, :]
    data_prot = data.loc[prot_rows, :]

    data_tscr = data_tscr.rename(columns=id_dict)
    data_prot = data_prot.rename(columns=id_dict)

    data_tscr.fillna(0, inplace=True)
    data_prot.fillna(0, inplace=True)

    adata_tscr = sc.AnnData(data_tscr)
    adata_prot = sc.AnnData(data_prot)

    adata_tscr.uns["iroot"] = 0  # random selection
    adata_prot.uns["iroot"] = 1448  # random selection
    # adata_prot.uns["iroot"] = 1319
    return adata_prot, adata_tscr

def remap_ensembl(adata: AnnData,
                  species: str = "human"):
    """
    Maps "ENSEMBL" genes to gene names using `mygene` package.
    Parameters
    ----------
    adata : AnnData
        AnnData object with gene names encoded by ENSEMBL
    species : str
        Species to use for converting the gene IDs.
    Returns
    -------

    """
    mg = mygene.MyGeneInfo()

    # List of Ensembl IDs
    ensembl_ids = adata.var_names  # ["ENSG00000139618", "ENSG00000284662"]

    # Query MyGene.info
    gene_info = mg.querymany(ensembl_ids, scopes="ensembl.gene", fields="symbol", species=species)

    # Extract results
    gene_map = {entry["query"]: entry.get("symbol", "NA") for entry in gene_info}
    new_var_names = [gene_map[v] for v in list(adata.var_names)]
    adata.var_names = new_var_names
    return

def load_lin2021_ngn2_timecourse():
    path = "data/lin2021_ngn2/E-MTAB-10632/matrices_timecourse/"
    counts_filepath = join(path, "counts.mtx")
    features_filename = join(path, "features.tsv")
    meta_filename = join(path, "meta.tsv")

    adata = sc.read_mtx(counts_filepath).T
    feats = pd.read_csv(features_filename, sep="\t", header=None)
    adata.var_names = list(feats[0].values)

    obs = pd.read_csv(meta_filename, sep="\t")
    adata.obs_names = obs.index
    adata.obs = obs

    # gene_coverage(adata)
    # retain_genes = adata.var["coverage"] > 0.5
    # adata = adata[:, retain_genes]
    print("Processing")
    sc.pp.filter_genes(adata, min_cells=100, inplace=True)
    sc.pp.filter_cells(adata, min_genes=100, inplace=True)
    sc.pp.filter_genes(adata, min_cells=100, inplace=True)
    sc.pp.log1p(adata)  # Take the natural log of the data.
    sc.pp.regress_out(adata, ['timepoint'])
    # sc.pp.combat(adata, key="batch", inplace=True)

    sc.pp.normalize_total(adata, inplace=True)  # Normalize to some total.

    sc.pp.scale(adata)  # rescale to 0 mean and variance of 1
    sc.pp.pca(adata, svd_solver="arpack")
    # sc.pp.neighbors(adata, n_neighbors=40, n_pcs=50)
    sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
    sc.tl.leiden(adata, resolution=1)
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)
    sc.tl.paga(adata)

    return adata


def load_lin2021_ngn2_timecourse_2():
    path = "data/lin2021_ngn2/E-MTAB-10632/matrices_timecourse/"
    counts_filepath = join(path, "counts.mtx")
    features_filename = join(path, "features.tsv")
    meta_filename = join(path, "meta.tsv")

    adata = sc.read_mtx(counts_filepath).T
    feats = pd.read_csv(features_filename, sep="\t", header=None)
    adata.var_names = list(feats[0].values)

    obs = pd.read_csv(meta_filename, sep="\t")
    adata.obs_names = obs.index
    adata.obs = obs

    # gene_coverage(adata)
    # retain_genes = adata.var["coverage"] > 0.5
    # adata = adata[:, retain_genes]
    print("Processing")
    sc.pp.filter_genes(adata, min_cells=100, inplace=True)
    sc.pp.filter_cells(adata, min_genes=100, inplace=True)
    sc.pp.filter_genes(adata, min_cells=100, inplace=True)
    sc.pp.log1p(adata)  # Take the natural log of the data.
    # sc.pp.regress_out(adata, ['timepoint'])
    # sc.pp.combat(adata, key="batch", inplace=True)

    sc.pp.normalize_total(adata, inplace=True)  # Normalize to some total.

    sc.pp.scale(adata)  # rescale to 0 mean and variance of 1
    sc.pp.pca(adata, svd_solver="arpack")
    # sc.pp.neighbors(adata, n_neighbors=40, n_pcs=50)
    sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
    sc.tl.leiden(adata, resolution=1)
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)
    sc.tl.paga(adata)

    return adata

def load_shan2024():
    path = "data/shan2024_in/GSM7814309_iP11N-d28-DM-Rep1.mtx"
    data = pd.read_csv(path, sep="\t").T
    adata = sc.AnnData(data)
    return adata


def load_cook2020(celltype: Literal["OVCA420", "A549", "DU145", "MCF7"] = "A549",
                  condition: Literal["TNF", "TGFB1", "EGF"] = "TNF",
                  path: str = "data/cook2020/"):
    file_prefix = join(path, celltype, f"GSE147405_{celltype}_{condition}_TimeCourse_")
    data_file = file_prefix + "UMI_matrix.csv.gz"
    metadata_file = file_prefix + "metadata.csv.gz"
    if not os.path.exists(data_file) or not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Data file {data_file} or metadata file {metadata_file} not found.")

    data = pd.read_csv(data_file, index_col=0, compression="gzip")
    metadata = pd.read_csv(metadata_file, index_col=0, compression="gzip")
    if data.shape[0] != metadata.shape[0] and data.shape[1] == metadata.shape[0]:
        data = data.T
    return sc.AnnData(data, obs=metadata)


def load_astral(filepath: str = "data/astral-scp-report.pg_matrix.tsv"):
    df = pd.read_csv(filepath, index_col=2, sep='\t')
    df = df.iloc[:, 3:]  # Drop the first three columns
    # Check the structure of the AnnData object
    df.fillna(0, inplace=True)
    return sc.AnnData(df.T)


def load_retinal(filepath: str = "data/retinal/cell-cycle-analysis/data/CR07_TMT16plex_FullProt_Proteins.tdt"):
    mapping0 = {
        "Palbo": "Abundance: 126, 1, 0",
        "Late G1_1": "Abundance: 127N, 1, 2",
        "G1/S": "Abundance: 127C, 1, 4",
        "S": "Abundance: 128N, 1, 6",
        "S/G2": "Abundance: 128C, 1, 8",
        "G2_2": "Abundance: 129N, 1, 10",
        "G2/M_1": "Abundance: 129C, 1, 12",
        "M/Early G1": "Abundance: 130N, 1, 15"
    }
    mapping1 = {
        "Palbo": "Abundance: 130C, 2, 0",
        "Late G1_1": "Abundance: 131N, 2, 2",
        "G1/S": "Abundance: 131C, 2, 4",
        "S": "Abundance: 132N, 2, 6",
        "S/G2": "Abundance: 132C, 2, 8",
        "G2_2": "Abundance: 133N, 2, 10",
        "G2/M_1": "Abundance: 133C, 2, 12",
        "M/Early G1": "Abundance: 134N, 2, 15"
    }

    imap = {}
    for k, v in mapping0.items():
        imap[v] = k
    for k, v in mapping1.items():
        imap[v] = k
    phase_ordinal_map = {
        "Palbo": 0,
        "Late G1_1": 1,
        "G1/S": 2,
        "S": 3,
        "S/G2": 4,
        "G2_2": 5,
        "G2/M_1": 6,
        "M/Early G1": 7}

    data = pd.read_csv(filepath, sep="\t", index_col="Gene Symbol")
    cols = data.columns
    abund_col = [c for c in cols if c.startswith("Abundance:")]
    prot_data = data[abund_col].T
    prot_data.fillna(0, inplace=True)
    obs = pd.DataFrame(index=prot_data.index)
    phase = []
    for idx in list(prot_data.index):
        phase.append(imap[idx])
    obs["phase"] = phase
    obs["phase_ordinal"] = obs["phase"].apply(lambda x: phase_ordinal_map[x])

    adata = ad.AnnData(prot_data, obs)
    adata.var_names_make_unique()
    return adata