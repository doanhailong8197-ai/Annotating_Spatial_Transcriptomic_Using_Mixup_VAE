# preprocess.py
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional, List


class TranscriptomicsDataset(Dataset):
    """
    Dataset lưu trữ biểu hiện gen đã chuẩn hóa Z-score theo miền (.data),
    ma trận log-normalized (.raw_counts / log_expr), kích thước thư viện (.library_size),
    nhãn loại tế bào (.labels) và tọa độ không gian (.coords).
    """
    def __init__(
        self,
        norm_data: np.ndarray,
        raw_counts: np.ndarray,
        library_size: np.ndarray,
        labels: np.ndarray,
        coords: Optional[np.ndarray] = None,
        cell_ids: Optional[List[str]] = None,
        class_counts: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.data = torch.tensor(norm_data, dtype=torch.float32)
        self.raw_counts = torch.tensor(raw_counts, dtype=torch.float32)
        self.library_size = torch.tensor(library_size, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

        if coords is not None:
            self.coords = torch.tensor(coords, dtype=torch.float32)
        else:
            self.coords = torch.zeros((norm_data.shape[0], 2), dtype=torch.float32)

        self.cell_ids = cell_ids if cell_ids is not None else [str(i) for i in range(norm_data.shape[0])]
        self.class_counts = (
            torch.tensor(class_counts, dtype=torch.float32)
            if class_counts is not None
            else None
        )

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.data[idx],
            "raw_counts": self.raw_counts[idx],
            "library_size": self.library_size[idx],
            "label": self.labels[idx],
            "coords": self.coords[idx],
            "idx": torch.tensor(idx, dtype=torch.long),
        }


def _to_dense_array(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _extract_raw_counts(adata: ad.AnnData) -> np.ndarray:
    if "counts" in adata.layers:
        mat = _to_dense_array(adata.layers["counts"])
    elif adata.raw is not None and adata.raw.shape[1] == adata.shape[1]:
        mat = _to_dense_array(adata.raw.X)
    else:
        mat = _to_dense_array(adata.X)

    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = np.clip(mat, a_min=0.0, a_max=None)

    if mat.max() < 30.0 and not np.allclose(mat, np.round(mat), atol=1e-2):
        mat = np.expm1(mat)

    return mat


def _extract_spatial_coords(adata_spa: ad.AnnData) -> np.ndarray:
    if "spatial" in adata_spa.obsm:
        coords = np.asarray(adata_spa.obsm["spatial"], dtype=np.float32)[:, :2]
    elif "X_spatial" in adata_spa.obsm:
        coords = np.asarray(adata_spa.obsm["X_spatial"], dtype=np.float32)[:, :2]
    else:
        coord_cols = None
        for pair in [("x", "y"), ("X", "Y"), ("row", "col"), ("array_row", "array_col"), ("imagecol", "imagerow")]:
            if pair[0] in adata_spa.obs.columns and pair[1] in adata_spa.obs.columns:
                coord_cols = list(pair)
                break
        if coord_cols is not None:
            coords = adata_spa.obs[coord_cols].to_numpy(dtype=np.float32)
        else:
            coords = np.zeros((adata_spa.n_obs, 2), dtype=np.float32)

    if coords.shape[0] > 0:
        c_min = coords.min(axis=0, keepdims=True)
        c_max = coords.max(axis=0, keepdims=True)
        denom = np.where((c_max - c_min) > 1e-8, c_max - c_min, 1.0)
        coords = (coords - c_min) / denom
    return coords


def _normalize_and_domain_standardize(
    raw_counts: np.ndarray, target_sum: float = 1e4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chuẩn hóa khử lệch miền (Domain-Specific Gene Standardization):
    1) Chuẩn hóa tổng đếm mỗi tế bào về target_sum (10^4) và biến đổi log1p.
    2) Chuẩn hóa Z-score (mean=0, std=1) theo từng gen trong nội bộ mỗi miền để
       khử sự khác biệt về độ nhạy đầu dò giữa scRNA-seq và ST (seqFISH/Visium).
    3) Chuẩn hóa L2 theo từng tế bào hoặc cắt ngưỡng [-5, 5] để ổn định gradient.
    """
    cell_sums = raw_counts.sum(axis=1, keepdims=True)
    log_lib_size = np.log(np.clip(cell_sums.squeeze(-1), a_min=1.0, a_max=None)).astype(np.float32)

    safe_sums = np.where(cell_sums > 0, cell_sums, 1.0)
    norm_counts = (raw_counts / safe_sums) * target_sum
    log_data = np.log1p(norm_counts).astype(np.float32)

    # Chuẩn hóa Z-score theo từng gen trong miền hiện tại
    gene_mean = log_data.mean(axis=0, keepdims=True)
    gene_std = log_data.std(axis=0, keepdims=True)
    gene_std = np.where(gene_std > 1e-6, gene_std, 1.0)

    z_data = (log_data - gene_mean) / gene_std
    z_data = np.clip(z_data, a_min=-5.0, a_max=5.0).astype(np.float32)

    return z_data, log_data, log_lib_size


def preprocess_pipeline(
    adata_rna: ad.AnnData,
    adata_spa: ad.AnnData,
    label_col: str = "cell_type",
    n_top_genes: int = 2000,
    batch_size: int = 256,
    target_sum: float = 1e4,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    rna = adata_rna.copy()
    spa = adata_spa.copy()
    rna.var_names_make_unique()
    spa.var_names_make_unique()

    # 1. Lấy tập gen chung giữa scRNA-seq và ST
    common_genes = np.intersect1d(rna.var_names.values, spa.var_names.values)
    if len(common_genes) == 0:
        raise ValueError("Không tìm thấy gen chung nào giữa adata_rna và adata_spa!")

    rna = rna[:, common_genes].copy()
    spa = spa[:, common_genes].copy()

    rna_raw_all = _extract_raw_counts(rna)
    spa_raw_all = _extract_raw_counts(spa)

    non_zero_mask = (rna_raw_all.sum(axis=0) > 0) & (spa_raw_all.sum(axis=0) > 0)
    if non_zero_mask.sum() > 0:
        rna = rna[:, non_zero_mask].copy()
        spa = spa[:, non_zero_mask].copy()
        rna_raw_all = rna_raw_all[:, non_zero_mask]
        spa_raw_all = spa_raw_all[:, non_zero_mask]

    # 2. Lọc Highly Variable Genes nếu số gen chung > n_top_genes
    if n_top_genes is not None and rna.n_vars > n_top_genes:
        rna_temp = ad.AnnData(X=rna_raw_all.copy())
        rna_temp.var_names = rna.var_names.copy()
        sc.pp.normalize_total(rna_temp, target_sum=target_sum)
        sc.pp.log1p(rna_temp)
        try:
            sc.pp.highly_variable_genes(rna_temp, n_top_genes=n_top_genes, subset=False)
            hvg_mask = rna_temp.var["highly_variable"].values
            if hvg_mask.sum() != n_top_genes:
                gene_vars = np.var(_to_dense_array(rna_temp.X), axis=0)
                top_idx = np.argsort(gene_vars)[::-1][:n_top_genes]
                hvg_mask = np.zeros(rna.n_vars, dtype=bool)
                hvg_mask[top_idx] = True
        except Exception:
            gene_vars = np.var(_to_dense_array(rna_temp.X), axis=0)
            top_idx = np.argsort(gene_vars)[::-1][:n_top_genes]
            hvg_mask = np.zeros(rna.n_vars, dtype=bool)
            hvg_mask[top_idx] = True

        rna = rna[:, hvg_mask].copy()
        spa = spa[:, hvg_mask].copy()
        rna_raw = rna_raw_all[:, hvg_mask]
        spa_raw = spa_raw_all[:, hvg_mask]
    else:
        rna_raw = rna_raw_all
        spa_raw = spa_raw_all

    # 3. Chuẩn hóa Z-score theo từng gen trên mỗi miền
    rna_z, rna_log, rna_lib = _normalize_and_domain_standardize(rna_raw, target_sum=target_sum)
    spa_z, spa_log, spa_lib = _normalize_and_domain_standardize(spa_raw, target_sum=target_sum)

    # 4. Mã hóa nhãn
    if label_col not in rna.obs.columns:
        raise KeyError(f"Cột nhãn '{label_col}' không tồn tại trong adata_rna.obs!")

    rna_labels_str = rna.obs[label_col].astype(str).values
    unique_labels = sorted(np.unique(rna_labels_str).tolist())
    label_mapping: Dict[str, int] = {name: idx for idx, name in enumerate(unique_labels)}

    rna_labels = np.array([label_mapping[name] for name in rna_labels_str], dtype=np.int64)
    class_counts = np.bincount(rna_labels, minlength=len(label_mapping)).astype(np.float32)

    if label_col in spa.obs.columns:
        spa_labels_str = spa.obs[label_col].astype(str).values
        spa_labels = np.array(
            [label_mapping.get(name, -1) for name in spa_labels_str], dtype=np.int64
        )
    else:
        spa_labels = np.full(spa.n_obs, -1, dtype=np.int64)

    spa_coords = _extract_spatial_coords(spa)
    rna_coords = np.zeros((rna.n_obs, 2), dtype=np.float32)

    # Lưu rna_z vào .data và rna_z vào .raw_counts để Decoder tái tạo đồng nhất không gian chuẩn hóa
    ds_rna = TranscriptomicsDataset(
        norm_data=rna_z,
        raw_counts=rna_log,
        library_size=rna_lib,
        labels=rna_labels,
        coords=rna_coords,
        cell_ids=rna.obs_names.tolist(),
        class_counts=class_counts,
    )
    ds_spa = TranscriptomicsDataset(
        norm_data=spa_z,
        raw_counts=spa_log,
        library_size=spa_lib,
        labels=spa_labels,
        coords=spa_coords,
        cell_ids=spa.obs_names.tolist(),
        class_counts=class_counts,
    )

    dl_rna = DataLoader(
        ds_rna,
        batch_size=min(batch_size, len(ds_rna)),
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    dl_spa = DataLoader(
        ds_spa,
        batch_size=min(batch_size, len(ds_spa)),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return dl_rna, dl_spa, label_mapping
