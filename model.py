# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


# ==============================================================================
# 1. SHARED-AFFINE DATASET-SPECIFIC BATCH NORMALIZATION (SA-DSBN)
# ==============================================================================
class SharedAffineDSBN1d(nn.Module):
    """
    Dataset-Specific Batch Normalization với tham số Affine (gamma, beta) DÙNG CHUNG:
    - Mỗi miền (scRNA-seq: domain_id=0, Spatial: domain_id=1) tự ước lượng running_mean
      và running_var riêng để loại bỏ hoàn toàn độ lệch tâm (mean shift) và lệch phương sai.
    - Sau khi chuẩn hóa về N(0, 1), cả 2 miền đi qua CÙNG một cặp (weight, bias) để
      đảm bảo đám mây điểm tiềm ẩn của scRNA-seq và ST luôn nằm chồng khít lên nhau!
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.bn_rna = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=False)
        self.bn_spa = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=False)

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, domain_id: int = 0, freeze_stats: bool = False) -> torch.Tensor:
        bn_layer = self.bn_rna if domain_id == 0 else self.bn_spa
        if freeze_stats and bn_layer.training:
            out = F.batch_norm(
                x,
                bn_layer.running_mean,
                bn_layer.running_var,
                None,
                None,
                training=False,
                momentum=0.0,
                eps=bn_layer.eps,
            )
        else:
            out = bn_layer(x)

        if self.affine:
            out = out * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return out


# ==============================================================================
# 2. VARIATIONAL ENCODER & DOMAIN-ADAPTIVE DECODER
# ==============================================================================
class VariationalEncoder(nn.Module):
    """
    Bộ mã hóa biến phân sử dụng SharedAffineDSBN1d ở cả các tầng ẩn lẫn đầu ra mu_head,
    giữ cho biểu diễn tiềm ẩn của scRNA-seq và ST hoàn toàn đồng tâm trên mọi chiều.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        h1, h2 = hidden_dims
        self.fc1 = nn.Linear(input_dim, h1)
        self.dsbn1 = SharedAffineDSBN1d(h1)
        self.act1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(h1, h2)
        self.dsbn2 = SharedAffineDSBN1d(h2)
        self.act2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)

        self.mu_fc = nn.Linear(h2, latent_dim)
        # Chuẩn hóa DSBN ngay trên mu để triệt tiêu độ dịch chuyển đám mây ST so với scRNA-seq
        self.mu_dsbn = SharedAffineDSBN1d(latent_dim, affine=True)
        self.logvar_head = nn.Linear(h2, latent_dim)

    def forward(
        self, x: torch.Tensor, domain_id: int = 0, freeze_stats: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.fc1(x)
        h = self.dsbn1(h, domain_id=domain_id, freeze_stats=freeze_stats)
        h = self.drop1(self.act1(h))

        h = self.fc2(h)
        h = self.dsbn2(h, domain_id=domain_id, freeze_stats=freeze_stats)
        h = self.drop2(self.act2(h))

        mu = self.mu_dsbn(self.mu_fc(h), domain_id=domain_id, freeze_stats=freeze_stats)
        logvar = torch.clamp(self.logvar_head(h), min=-6.0, max=2.0)
        std = torch.exp(0.5 * logvar)

        if self.training and not freeze_stats:
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return z, mu, logvar


class DomainAdaptiveDecoder(nn.Module):
    """
    Bộ giải mã thích ứng miền (sử dụng SharedAffineDSBN1d):
    - Cho phép tái tạo chính xác biểu hiện gen chuẩn hóa của từng miền mà KHÔNG ép
      vector tiềm ẩn z phải lưu trữ thông tin khác biệt công nghệ (seqFISH vs scRNA-seq).
    - Khớp 100% không gian đầu vào của Encoder khi chạy vòng lặp Manifold Rectifier.
    """
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, int] = (128, 256),
    ):
        super().__init__()
        h1, h2 = hidden_dims
        self.fc1 = nn.Linear(latent_dim, h1)
        self.dsbn1 = SharedAffineDSBN1d(h1)
        self.act1 = nn.ELU()

        self.fc2 = nn.Linear(h1, h2)
        self.dsbn2 = SharedAffineDSBN1d(h2)
        self.act2 = nn.ELU()

        self.out_head = nn.Linear(h2, output_dim)

    def forward(
        self, z: torch.Tensor, domain_id: int = 0, freeze_stats: bool = False
    ) -> torch.Tensor:
        h = self.fc1(z)
        h = self.dsbn1(h, domain_id=domain_id, freeze_stats=freeze_stats)
        h = self.act1(h)

        h = self.fc2(h)
        h = self.dsbn2(h, domain_id=domain_id, freeze_stats=freeze_stats)
        h = self.act2(h)

        x_hat = self.out_head(h)
        return x_hat

    def decode_to_surrogate_input(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z, domain_id=0, freeze_stats=True)


def profile_reconstruction_loss(x_true: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Kết hợp Smooth-L1 (Huber) và Cosine Profile Similarity trên biểu hiện gen chuẩn hóa."""
    huber = F.smooth_l1_loss(x_hat, x_true, reduction="mean")
    cos_loss = 1.0 - F.cosine_similarity(x_hat, x_true, dim=-1).mean()
    return huber + 0.5 * cos_loss


# ==============================================================================
# 3. BỘ TĂNG CƯỜNG LATENT MIXUP & CHỈNH LƯU ĐA TẠP (NON-GMM, NON-KNN)
# ==============================================================================
class LatentMixupRectifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        latent_dim: int,
        class_counts: Optional[torch.Tensor] = None,
        queue_size_per_class: int = 24,
        beta_alpha: float = 2.0,
        cos_threshold: float = 0.35,
        rect_blend: float = 0.35,
        trust_eta: float = 1.5,
        imbalance_power: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.queue_size = queue_size_per_class
        self.beta_alpha = beta_alpha
        self.cos_threshold = cos_threshold
        self.rect_blend = rect_blend
        self.trust_eta = trust_eta

        if class_counts is not None:
            counts = class_counts.float().clamp(min=1.0)
            n_max = counts.max()
            rarity = 1.0 - (counts / n_max).pow(imbalance_power)
            rarity = torch.clamp(rarity, min=0.10, max=1.0)
        else:
            rarity = torch.ones(num_classes) * 0.5
            counts = torch.ones(num_classes)

        self.register_buffer("class_counts", counts)
        self.register_buffer("class_rarity", rarity)

        self.register_buffer("mu_queue", torch.zeros(num_classes, queue_size_per_class, latent_dim))
        self.register_buffer("logvar_queue", torch.zeros(num_classes, queue_size_per_class, latent_dim))
        self.register_buffer("queue_ptr", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("queue_filled", torch.zeros(num_classes, dtype=torch.long))

    @torch.no_grad()
    def update_queues(self, mu: torch.Tensor, logvar: torch.Tensor, labels: torch.Tensor):
        for c in torch.unique(labels):
            c_idx = int(c.item())
            if c_idx < 0 or c_idx >= self.num_classes:
                continue
            mask = labels == c
            mu_c = mu[mask].detach()
            lv_c = logvar[mask].detach()
            for i in range(mu_c.shape[0]):
                ptr = int(self.queue_ptr[c_idx].item())
                self.mu_queue[c_idx, ptr] = mu_c[i]
                self.logvar_queue[c_idx, ptr] = lv_c[i]
                self.queue_ptr[c_idx] = (ptr + 1) % self.queue_size
                self.queue_filled[c_idx] = min(self.queue_size, int(self.queue_filled[c_idx].item()) + 1)

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        labels: torch.Tensor,
        encoder: VariationalEncoder,
        decoder: DomainAdaptiveDecoder,
    ) -> Dict[str, torch.Tensor]:
        device = mu.device
        self.update_queues(mu, logvar, labels)

        sample_rarity = self.class_rarity[labels]
        keep_prob = torch.clamp(sample_rarity * 1.2, min=0.15, max=1.0)
        selected_mask = torch.bernoulli(keep_prob).bool()
        if selected_mask.sum() < 2:
            selected_mask = torch.ones_like(labels, dtype=torch.bool)

        anchor_mu = mu[selected_mask]
        anchor_logvar = logvar[selected_mask]
        anchor_std = torch.exp(0.5 * anchor_logvar)
        anchor_labels = labels[selected_mask]
        M = anchor_mu.shape[0]

        partner_mu = torch.zeros_like(anchor_mu)
        partner_std = torch.zeros_like(anchor_std)

        for i in range(M):
            c_idx = int(anchor_labels[i].item())
            filled = int(self.queue_filled[c_idx].item())
            if filled > 1:
                cand_mu = self.mu_queue[c_idx, :filled]
                cand_std = torch.exp(0.5 * self.logvar_queue[c_idx, :filled])
                rand_idx = torch.randint(0, filled, (1,), device=device).item()
                partner_mu[i] = cand_mu[rand_idx]
                partner_std[i] = cand_std[rand_idx]
            else:
                partner_mu[i] = anchor_mu[i]
                partner_std[i] = anchor_std[i]

        cos_sim = F.cosine_similarity(anchor_mu, partner_mu, dim=-1, eps=1e-8)
        coherence_gate = (cos_sim >= self.cos_threshold).float().unsqueeze(-1)

        beta_dist = torch.distributions.Beta(self.beta_alpha, self.beta_alpha)
        raw_lam = beta_dist.sample((M, 1)).to(device)
        raw_lam = torch.max(raw_lam, 1.0 - raw_lam)

        local_lam = 0.90 + 0.08 * torch.rand((M, 1), device=device)
        lam_intra = coherence_gate * raw_lam + (1.0 - coherence_gate) * local_lam

        mu_mix_intra = lam_intra * anchor_mu + (1.0 - lam_intra) * partner_mu
        std_mix_intra = lam_intra * anchor_std + (1.0 - lam_intra) * partner_std
        z_mix_intra = mu_mix_intra + 0.5 * torch.randn_like(std_mix_intra) * std_mix_intra

        # Boundary Remix nhẹ bám sát biên lớp hiếm
        norm_anchor = F.normalize(anchor_mu.detach(), dim=-1)
        norm_batch = F.normalize(mu.detach(), dim=-1)
        sim_matrix = torch.matmul(norm_anchor, norm_batch.t())
        same_class_mask = anchor_labels.unsqueeze(1) == labels.unsqueeze(0)
        sim_matrix = sim_matrix.masked_fill(same_class_mask, -1e9)
        hard_neg_idx = sim_matrix.argmax(dim=1)

        neg_mu = mu[hard_neg_idx].detach()
        lam_bound = 0.82 + 0.14 * torch.rand((M, 1), device=device)
        z_mix_bound = lam_bound * anchor_mu + (1.0 - lam_bound) * neg_mu + 0.3 * torch.randn_like(anchor_std) * anchor_std

        z_mix_all = torch.cat([z_mix_intra, z_mix_bound], dim=0)
        y_mix_all = torch.cat([anchor_labels, anchor_labels], dim=0)

        # Chỉnh lưu Đa tạp qua Decoder -> Encoder (cùng miền chuẩn hóa domain_id=0)
        x_surrogate = decoder.decode_to_surrogate_input(z_mix_all)
        _, z_rect_all, _ = encoder(x_surrogate, domain_id=0, freeze_stats=True)

        z_star = (1.0 - self.rect_blend) * z_mix_all + self.rect_blend * z_rect_all

        discrepancy = torch.mean((z_mix_all.detach() - z_rect_all.detach()) ** 2, dim=-1)
        trust_weights = torch.exp(-self.trust_eta * discrepancy)

        z_rect_intra = z_rect_all[:M]
        valid_weights = coherence_gate.squeeze(-1) + 0.2 * (1.0 - coherence_gate.squeeze(-1))
        rect_loss = torch.mean(valid_weights * torch.mean((z_mix_intra.detach() - z_rect_intra) ** 2, dim=-1))

        return {
            "z_aug": z_star,
            "y_aug": y_mix_all,
            "trust_weights": trust_weights,
            "rect_loss": rect_loss,
        }


# ==============================================================================
# 4. MINIBATCH UNBALANCED OPTIMAL TRANSPORT (CHỐNG COLLAPSE ĐƠN LỚP)
# ==============================================================================
class MinibatchUOT(nn.Module):
    """
    Sinkhorn Unbalanced Optimal Transport với điều chuẩn biên cột đủ chặt (reg_m_kl_col)
    để ngăn chặn hiện tượng dồn toàn bộ tế bào không gian vào một lớp duy nhất.
    """
    def __init__(
        self,
        reg_ent: float = 0.06,
        reg_m_kl_row: float = 0.5,
        reg_m_kl_col: float = 1.5,
        num_iters: int = 25,
    ):
        super().__init__()
        self.reg_ent = reg_ent
        self.reg_m_kl_row = reg_m_kl_row
        self.reg_m_kl_col = reg_m_kl_col
        self.num_iters = num_iters

    @torch.no_grad()
    def compute_transport_plan(
        self, cost_matrix: torch.Tensor, target_prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B_sp, C = cost_matrix.shape
        device = cost_matrix.device

        a = torch.full((B_sp,), 1.0 / B_sp, device=device)
        if target_prior is not None:
            b = target_prior / (target_prior.sum() + 1e-8)
        else:
            b = torch.full((C,), 1.0 / C, device=device)

        log_a = torch.log(a + 1e-8)
        log_b = torch.log(b + 1e-8)
        log_K = -cost_matrix / self.reg_ent

        tau_row = self.reg_m_kl_row / (self.reg_m_kl_row + self.reg_ent)
        tau_col = self.reg_m_kl_col / (self.reg_m_kl_col + self.reg_ent)

        u = torch.zeros(B_sp, device=device)
        v = torch.zeros(C, device=device)

        for _ in range(self.num_iters):
            u = tau_row * (log_a - torch.logsumexp(log_K + v.unsqueeze(0), dim=1))
            v = tau_col * (log_b - torch.logsumexp(log_K + u.unsqueeze(1), dim=0))

        log_T = u.unsqueeze(1) + log_K + v.unsqueeze(0)
        return torch.exp(log_T)


# ==============================================================================
# 5. KIẾN TRÚC MaRe-VAE HOÀN CHỈNH (ĐÃ KHẮC PHỤC LỆCH MIỀN & COLLAPSE)
# ==============================================================================
class MaReVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (256, 128),
        class_counts: Optional[torch.Tensor] = None,
        cosine_scale: float = 16.0,
        logit_adj_tau: float = 0.0,
        beta_kl: float = 0.005,
        lambda_cls: float = 2.0,
        lambda_aug: float = 1.0,
        lambda_rect: float = 0.5,
        lambda_uot: float = 0.4,
        lambda_spa: float = 0.2,
        open_set_keep_ratio: float = 0.65,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.cosine_scale = cosine_scale
        self.logit_adj_tau = logit_adj_tau
        self.open_set_keep_ratio = open_set_keep_ratio

        self.beta_kl = beta_kl
        self.lambda_cls = lambda_cls
        self.lambda_aug = lambda_aug
        self.lambda_rect = lambda_rect
        self.lambda_uot = lambda_uot
        self.lambda_spa = lambda_spa

        self.encoder = VariationalEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        self.decoder = DomainAdaptiveDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=(hidden_dims[1], hidden_dims[0]),
        )

        self.prototypes = nn.Parameter(torch.randn(num_classes, latent_dim))
        nn.init.orthogonal_(self.prototypes)

        # Trọng số cân bằng lớp nhẹ cho hàm Cross-Entropy trên scRNA-seq để lớp đa số
        # (như ExE endoderm) không lấn át các lớp hiếm
        if class_counts is not None:
            counts = class_counts.float().clamp(min=1.0)
            cb_weights = (1.0 / counts).pow(0.25)
            cb_weights = cb_weights / cb_weights.mean()
        else:
            cb_weights = torch.ones(num_classes)
        self.register_buffer("ce_class_weights", cb_weights)

        # Tiên nghiệm đích trên miền Spatial được ước lượng động bằng EMA (khởi tạo ĐỒNG ĐỀU 1/C,
        # tuyệt đối không dùng tỷ lệ lệch của scRNA-seq áp đặt lên ST!)
        self.register_buffer("spa_ema_prior", torch.full((num_classes,), 1.0 / num_classes))

        self.mixup_rectifier = LatentMixupRectifier(
            num_classes=num_classes,
            latent_dim=latent_dim,
            class_counts=class_counts,
        )

        self.uot = MinibatchUOT(reg_ent=0.06, reg_m_kl_row=0.5, reg_m_kl_col=1.5, num_iters=25)

    def compute_logits(self, z: torch.Tensor, apply_logit_adj: bool = False) -> torch.Tensor:
        z_norm = F.normalize(z, dim=-1)
        w_norm = F.normalize(self.prototypes, dim=-1)
        return self.cosine_scale * torch.matmul(z_norm, w_norm.t())

    @staticmethod
    def kl_divergence_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

    def prototype_anchor_loss(self, mu_rna: torch.Tensor, y_rna: torch.Tensor) -> torch.Tensor:
        """
        Khóa chặt mỗi vector nguyên mẫu self.prototypes[c] vào đúng tâm thực tế
        của các tế bào thuộc lớp c trong không gian tiềm ẩn.
        """
        w_norm = F.normalize(self.prototypes, dim=-1)
        loss = torch.tensor(0.0, device=mu_rna.device)
        unique_classes = torch.unique(y_rna)
        for c in unique_classes:
            c_idx = int(c.item())
            if c_idx < 0 or c_idx >= self.num_classes:
                continue
            mean_c = F.normalize(mu_rna[y_rna == c].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
            loss = loss + (1.0 - torch.sum(mean_c * w_norm[c_idx]))
        return loss / max(1, len(unique_classes))

    @staticmethod
    def coral_alignment_loss(mu_rna: torch.Tensor, mu_spa: torch.Tensor) -> torch.Tensor:
        """
        Deep CORAL Loss: Căn chỉnh trung bình và ma trận hiệp phương sai giữa scRNA-seq và ST.
        """
        if mu_rna.shape[0] <= 2 or mu_spa.shape[0] <= 2:
            return torch.tensor(0.0, device=mu_rna.device)
        d = mu_rna.shape[1]
        mean_r = mu_rna.mean(dim=0, keepdim=True)
        mean_s = mu_spa.mean(dim=0, keepdim=True)
        mean_loss = F.mse_loss(mean_r, mean_s)

        cent_r = mu_rna - mean_r
        cent_s = mu_spa - mean_s
        cov_r = torch.matmul(cent_r.t(), cent_r) / (mu_rna.shape[0] - 1)
        cov_s = torch.matmul(cent_s.t(), cent_s) / (mu_spa.shape[0] - 1)
        cov_loss = torch.sum((cov_r - cov_s) ** 2) / (4.0 * d * d)
        return mean_loss + cov_loss

    def spatial_smoothness_loss(
        self, probs_spa: torch.Tensor, coords_spa: torch.Tensor, sigma_coord: float = 0.06
    ) -> torch.Tensor:
        if coords_spa.shape[0] <= 1 or torch.all(coords_spa == 0):
            return torch.tensor(0.0, device=probs_spa.device)

        dist_sq = torch.cdist(coords_spa, coords_spa, p=2).pow(2)
        weights = torch.exp(-dist_sq / (2.0 * (sigma_coord ** 2)))
        weights.fill_diagonal_(0.0)
        row_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weights = weights / row_sum

        prob_diff_sq = torch.cdist(probs_spa, probs_spa, p=2).pow(2)
        return torch.mean(weights * prob_diff_sq)

    def forward_step(
        self,
        batch_rna: Dict[str, torch.Tensor],
        batch_spa: Optional[Dict[str, torch.Tensor]] = None,
        use_latent_mixup: bool = True,
    ) -> Dict[str, torch.Tensor]:
        x_rna = batch_rna["x"]
        y_rna = batch_rna["label"]

        # 1. Nhánh scRNA-seq tham chiếu
        z_rna, mu_rna, logvar_rna = self.encoder(x_rna, domain_id=0, freeze_stats=False)
        x_hat_rna = self.decoder(z_rna, domain_id=0, freeze_stats=False)

        recon_loss_rna = profile_reconstruction_loss(x_rna, x_hat_rna)
        kl_loss_rna = self.kl_divergence_normal(mu_rna, logvar_rna)

        logits_rna = self.compute_logits(mu_rna)
        cls_loss_real = F.cross_entropy(logits_rna, y_rna, weight=self.ce_class_weights)
        anchor_loss = self.prototype_anchor_loss(mu_rna, y_rna)

        total_loss = (
            recon_loss_rna
            + self.beta_kl * kl_loss_rna
            + self.lambda_cls * (cls_loss_real + 0.5 * anchor_loss)
        )

        # 2. Nhánh Latent Mixup & Chỉnh lưu Đa tạp
        aug_cls_loss = torch.tensor(0.0, device=x_rna.device)
        rect_loss = torch.tensor(0.0, device=x_rna.device)

        if use_latent_mixup:
            aug_out = self.mixup_rectifier(
                mu=mu_rna,
                logvar=logvar_rna,
                labels=y_rna,
                encoder=self.encoder,
                decoder=self.decoder,
            )
            z_aug = aug_out["z_aug"]
            y_aug = aug_out["y_aug"]
            trust_w = aug_out["trust_weights"]
            rect_loss = aug_out["rect_loss"]

            logits_aug = self.compute_logits(z_aug)
            per_sample_ce = F.cross_entropy(logits_aug, y_aug, weight=self.ce_class_weights, reduction="none")
            aug_cls_loss = torch.mean(trust_w * per_sample_ce)

            total_loss = total_loss + self.lambda_aug * aug_cls_loss + self.lambda_rect * rect_loss
        else:
            self.mixup_rectifier.update_queues(mu_rna, logvar_rna, y_rna)

        # 3. Nhánh Spatial Transcriptomics & Căn chỉnh Open-Set Safe UOT
        recon_loss_spa = torch.tensor(0.0, device=x_rna.device)
        uot_loss = torch.tensor(0.0, device=x_rna.device)
        spa_smooth_loss = torch.tensor(0.0, device=x_rna.device)

        if batch_spa is not None:
            x_spa = batch_spa["x"]
            coords_spa = batch_spa["coords"]

            z_spa, mu_spa, logvar_spa = self.encoder(x_spa, domain_id=1, freeze_stats=False)
            x_hat_spa = self.decoder(z_spa, domain_id=1, freeze_stats=False)

            recon_loss_spa = profile_reconstruction_loss(x_spa, x_hat_spa)
            kl_loss_spa = self.kl_divergence_normal(mu_spa, logvar_spa)
            coral_loss = self.coral_alignment_loss(mu_rna, mu_spa)

            total_loss = total_loss + recon_loss_spa + self.beta_kl * kl_loss_spa + 0.5 * coral_loss

            # Chỉ kích hoạt UOT pseudo-label alignment sau giai đoạn Warm-up (khi use_latent_mixup=True)
            # để tránh việc ghép nhầm vào các nguyên mẫu chưa hội tụ ở những epoch đầu
            if use_latent_mixup:
                z_spa_norm = F.normalize(mu_spa, dim=-1)
                w_norm = F.normalize(self.prototypes.detach(), dim=-1)
                cos_sim_spa = torch.matmul(z_spa_norm, w_norm.t())
                cost_matrix = 1.0 - cos_sim_spa  # (B_sp, C)

                # Chỉ lấy các tế bào ST có độ tương đồng Cosine cao (Known candidates) để cập nhật prior & UOT
                max_cos, _ = cos_sim_spa.detach().max(dim=-1)
                cos_cutoff = torch.quantile(max_cos, q=1.0 - self.open_set_keep_ratio)
                known_cand_mask = (max_cos >= cos_cutoff).float()

                transport_plan = self.uot.compute_transport_plan(
                    cost_matrix.detach(), target_prior=self.spa_ema_prior
                )
                pseudo_probs = transport_plan / (transport_plan.sum(dim=1, keepdim=True) + 1e-8)

                # Cập nhật EMA prior trên miền ST nhưng giữ 30% hỗn hợp đồng đều để không bao giờ sụp đổ về 1 lớp
                with torch.no_grad():
                    batch_est_prior = (pseudo_probs * known_cand_mask.unsqueeze(1)).sum(dim=0)
                    batch_est_prior = batch_est_prior / (batch_est_prior.sum() + 1e-8)
                    uniform_p = torch.full_like(batch_est_prior, 1.0 / self.num_classes)
                    blended_p = 0.7 * batch_est_prior + 0.3 * uniform_p
                    self.spa_ema_prior.mul_(0.95).add_(0.05 * blended_p)

                logits_spa = self.compute_logits(mu_spa)
                log_probs_spa = F.log_softmax(logits_spa, dim=-1)
                uot_ce = -torch.sum(pseudo_probs.detach() * log_probs_spa, dim=-1)

                # Chỉ kéo các tế bào Known candidate về phía Prototype, giữ nguyên các tế bào Unknown ở vùng đệm
                uot_loss = torch.sum(known_cand_mask * uot_ce) / (known_cand_mask.sum() + 1e-6)

                probs_spa = F.softmax(logits_spa, dim=-1)
                spa_smooth_loss = self.spatial_smoothness_loss(probs_spa, coords_spa)

                total_loss = (
                    total_loss
                    + self.lambda_uot * uot_loss
                    + self.lambda_spa * spa_smooth_loss
                )

        return {
            "loss": total_loss,
            "recon_rna": recon_loss_rna.detach(),
            "recon_spa": recon_loss_spa.detach(),
            "cls_real": cls_loss_real.detach(),
            "cls_aug": aug_cls_loss.detach(),
            "rect_loss": rect_loss.detach(),
            "uot_loss": uot_loss.detach(),
            "spa_loss": spa_smooth_loss.detach(),
        }

    @torch.no_grad()
    def predict_dataloader(
        self,
        dataloader,
        domain_id: int = 1,
        detect_unknown: bool = False,
        unknown_quantile: float = 0.35,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Dự đoán nhãn trên toàn bộ DataLoader.
        Lưu ý: Đối với dữ liệu Spatial (domain_id=1), cập nhật thống kê BatchNorm toàn cục
        trước khi suy luận để loại bỏ hoàn toàn độ lệch giữa mini-batch và toàn tập.
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        all_mu = []
        all_logits = []
        all_probs = []
        all_labels = []
        all_scores = []

        w_norm = F.normalize(self.prototypes, dim=-1)

        for batch in dataloader:
            x = batch["x"].to(device)
            labels = batch["label"]
            _, mu, _ = self.encoder(x, domain_id=domain_id, freeze_stats=True)

            logits = self.compute_logits(mu)
            probs = F.softmax(logits, dim=-1)

            # Điểm tin cậy kết hợp giữa: (1) Tương đồng Cosine cực đại với Prototype,
            # (2) Xác suất Softmax cực đại, và (3) Âm Entropy phân lớp
            z_norm = F.normalize(mu, dim=-1)
            cos_sim = torch.matmul(z_norm, w_norm.t())
            max_cos, _ = cos_sim.max(dim=-1)
            max_prob, _ = probs.max(dim=-1)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1) / math.log(self.num_classes)
            known_score = 0.50 * max_cos + 0.30 * max_prob + 0.20 * (1.0 - entropy)

            all_mu.append(mu.cpu())
            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_scores.append(known_score.cpu())

        mu_cat = torch.cat(all_mu, dim=0)
        logits_cat = torch.cat(all_logits, dim=0)
        probs_cat = torch.cat(all_probs, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        scores_cat = torch.cat(all_scores, dim=0)

        preds = probs_cat.argmax(dim=-1)

        if detect_unknown:
            threshold = torch.quantile(scores_cat, q=unknown_quantile)
            preds = torch.where(scores_cat < threshold, torch.tensor(-1, dtype=torch.long), preds)

        return {
            "preds": preds,
            "probs": probs_cat,
            "logits": logits_cat,
            "latent": mu_cat,
            "labels": labels_cat,
            "known_scores": scores_cat,
        }
