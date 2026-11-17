# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


# ==============================================================================
# 1. DATASET-SPECIFIC BATCH NORMALIZATION (DSBN) CÓ CHẾ ĐỘ ĐÓNG BĂNG THỐNG KÊ
# ==============================================================================
class DSBN1d(nn.Module):
    """
    Dataset-Specific Batch Normalization cho 2 miền dữ liệu:
    - domain_id = 0: scRNA-seq reference
    - domain_id = 1: Spatial Transcriptomics query
    Hỗ trợ cờ `freeze_stats=True` khi chạy nhánh Chỉnh lưu Đa tạp (Manifold Rectifier)
    để các điểm Mixup nhân tạo không bao giờ làm lệch trung bình/phương sai của dữ liệu thực.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn_rna = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        self.bn_spa = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor, domain_id: int = 0, freeze_stats: bool = False) -> torch.Tensor:
        bn_layer = self.bn_rna if domain_id == 0 else self.bn_spa
        if freeze_stats and bn_layer.training:
            # Sử dụng running_mean và running_var đã học từ dữ liệu thực mà không cập nhật
            return F.batch_norm(
                x,
                bn_layer.running_mean,
                bn_layer.running_var,
                bn_layer.weight,
                bn_layer.bias,
                training=False,
                momentum=0.0,
                eps=bn_layer.eps,
            )
        return bn_layer(x)


# ==============================================================================
# 2. VARIATIONAL ENCODER & ZINB DECODER
# ==============================================================================
class VariationalEncoder(nn.Module):
    """
    Ánh xạ biểu hiện gen chuẩn hóa x -> phân phối hậu nghiệm q_phi(z | x) = N(mu, diag(sigma^2)).
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
        self.dsbn1 = DSBN1d(h1)
        self.act1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(h1, h2)
        self.dsbn2 = DSBN1d(h2)
        self.act2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)

        self.mu_head = nn.Linear(h2, latent_dim)
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

        mu = self.mu_head(h)
        logvar = torch.clamp(self.logvar_head(h), min=-8.0, max=4.0)
        std = torch.exp(0.5 * logvar)

        if self.training and not freeze_stats:
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return z, mu, logvar


class ZINBDecoder(nn.Module):
    """
    Bộ giải mã sinh học theo phân phối Zero-Inflated Negative Binomial (ZINB).
    Đồng thời cung cấp hàm `decode_to_surrogate_input(z)` khả vi hoàn toàn để
    ánh xạ vector tiềm ẩn z_mix về không gian đầu vào chuẩn hóa phục vụ Bộ chỉnh lưu Đa tạp.
    """
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, int] = (128, 256),
        target_sum: float = 1e4,
    ):
        super().__init__()
        h1, h2 = hidden_dims
        self.target_sum = target_sum

        self.net = nn.Sequential(
            nn.Linear(latent_dim, h1),
            nn.LayerNorm(h1),
            nn.ELU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.ELU(),
        )
        # Tần số biểu hiện tương đối rho_ng (tổng theo gen bằng 1)
        self.rho_head = nn.Sequential(
            nn.Linear(h2, output_dim),
            nn.Softmax(dim=-1),
        )
        # Logit xác suất dropout pi_ng
        self.pi_logits_head = nn.Linear(h2, output_dim)
        # Tham số phân tán theo từng gen (log-theta)
        self.log_theta = nn.Parameter(torch.zeros(output_dim))

        # Hệ số tỷ lệ khả học để khớp với bước batch-scale ở tiền xử lý
        self.surrogate_scale = nn.Parameter(torch.tensor(1.0 / math.log(target_sum + 1.0)))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(z)
        rho = self.rho_head(h)
        pi_logits = self.pi_logits_head(h)
        theta = F.softplus(self.log_theta) + 1e-4
        return rho, pi_logits, theta

    def decode_to_surrogate_input(self, z: torch.Tensor) -> torch.Tensor:
        """
        Tạo biểu hiện gen chuẩn hóa kỳ vọng (khả vi) từ vector tiềm ẩn z:
        x_hat = scale * ln(rho * 10^4 + 1)
        """
        rho, _, _ = self.forward(z)
        log_expr = torch.log1p(rho * self.target_sum)
        scale = torch.clamp(self.surrogate_scale, min=0.05, max=1.0)
        return log_expr * scale


def zinb_negative_log_likelihood(
    raw_counts: torch.Tensor,
    library_size: torch.Tensor,
    rho: torch.Tensor,
    pi_logits: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Tính hàm mất mát âm log-likelihood của phân phối ZINB."""
    lib_counts = torch.exp(torch.clamp(library_size, min=0.0, max=15.0))
    mu = rho * lib_counts + eps
    theta = theta.unsqueeze(0)

    t1 = torch.lgamma(raw_counts + theta + eps) - torch.lgamma(theta + eps) - torch.lgamma(raw_counts + 1.0)
    t2 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
    t3 = raw_counts * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    nb_log_prob = t1 + t2 + t3

    zero_nb = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
    zero_case = -F.softplus(-pi_logits + zero_nb) - F.softplus(pi_logits)
    non_zero_case = -F.softplus(pi_logits) + nb_log_prob

    mask = (raw_counts < eps).float()
    log_lik = mask * zero_case + (1.0 - mask) * non_zero_case
    return -log_lik.mean()


# ==============================================================================
# 3. BỘ TĂNG CƯỜNG LATENT MIXUP & CHỈNH LƯU ĐA TẠP (KHÔNG DÙNG GMM, KHÔNG DÙNG KNN)
# ==============================================================================
class LatentMixupRectifier(nn.Module):
    """
    Thực hiện Latent Posterior Mixup không cần GMM và không cần k-NN:
    1) Lưu một hàng đợi phân phối hậu nghiệm nhỏ (FIFO Class Anchor Bank) cho từng lớp
       để ngay cả các lớp cực hiếm (chỉ có vài tế bào) luôn có đối tác ghép cặp trong mỗi batch.
    2) Cổng tương đồng Cosine nội tại (Cosine Coherence Gate): Khi ghép hai tế bào cùng lớp hiếm,
       nếu góc Cosine giữa chúng thấp (thuộc 2 phân nhóm dị질 xa nhau như ở CAFs), hệ số trộn lambda
       tự động co về gần 1.0 để chỉ mở rộng phương sai cục bộ mà không nối chéo qua khoảng trống.
    3) Bộ Chỉnh lưu Đa tạp (Manifold Rectifier): Đưa z_mix qua Decoder -> Encoder để uốn cong
       về đa tạp sinh học và tính trọng số tin cậy w(z*).
    """
    def __init__(
        self,
        num_classes: int,
        latent_dim: int,
        class_counts: Optional[torch.Tensor] = None,
        queue_size_per_class: int = 16,
        beta_alpha: float = 2.0,
        cos_threshold: float = 0.35,
        rect_blend: float = 0.4,
        trust_eta: float = 2.0,
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

        # Tính trọng số ưu tiên tăng cường theo mức độ hiếm của từng lớp
        if class_counts is not None:
            counts = class_counts.float().clamp(min=1.0)
            n_max = counts.max()
            rarity = 1.0 - (counts / n_max).pow(imbalance_power)
            # Chuẩn hóa về [0.05, 1.0] (lớp đa số hầu như không cần sinh thêm)
            rarity = torch.clamp(rarity, min=0.0, max=1.0)
        else:
            rarity = torch.ones(num_classes) * 0.5
            counts = torch.ones(num_classes)

        self.register_buffer("class_counts", counts)
        self.register_buffer("class_rarity", rarity)

        # Bộ nhớ đệm FIFO lưu (mu, logvar) gần nhất của từng lớp (thay thế hoàn toàn k-NN toàn cục)
        self.register_buffer("mu_queue", torch.zeros(num_classes, queue_size_per_class, latent_dim))
        self.register_buffer("logvar_queue", torch.zeros(num_classes, queue_size_per_class, latent_dim))
        self.register_buffer("queue_ptr", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("queue_filled", torch.zeros(num_classes, dtype=torch.long))

    @torch.no_grad()
    def update_queues(self, mu: torch.Tensor, logvar: torch.Tensor, labels: torch.Tensor):
        """Cập nhật hàng đợi hậu nghiệm cho các lớp xuất hiện trong mini-batch hiện tại."""
        for c in torch.unique(labels):
            c_idx = int(c.item())
            if c_idx < 0 or c_idx >= self.num_classes:
                continue
            mask = labels == c
            mu_c = mu[mask].detach()
            lv_c = logvar[mask].detach()
            n_samples = mu_c.shape[0]

            for i in range(n_samples):
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
        decoder: ZINBDecoder,
    ) -> Dict[str, torch.Tensor]:
        """
        Sinh các điểm tiềm ẩn nhân tạo đã được chỉnh lưu qua cặp Decoder-Encoder.
        """
        device = mu.device
        self.update_queues(mu, logvar, labels)

        # Chọn các tế bào thuộc các lớp thiểu số trong batch làm neo (anchors)
        sample_rarity = self.class_rarity[labels]
        # Lấy mẫu xác suất theo độ hiếm để tập trung tài nguyên cho các lớp thiểu số
        keep_prob = torch.clamp(sample_rarity * 1.2, min=0.1, max=1.0)
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

        # 1. Ghép cặp nội lớp từ bộ nhớ đệm (Intra-Class Pairing)
        for i in range(M):
            c_idx = int(anchor_labels[i].item())
            filled = int(self.queue_filled[c_idx].item())
            if filled > 1:
                cand_mu = self.mu_queue[c_idx, :filled]
                cand_std = torch.exp(0.5 * self.logvar_queue[c_idx, :filled])
                # Chọn ngẫu nhiên 1 đối tác cùng lớp từ queue
                rand_idx = torch.randint(0, filled, (1,), device=device).item()
                partner_mu[i] = cand_mu[rand_idx]
                partner_std[i] = cand_std[rand_idx]
            else:
                partner_mu[i] = anchor_mu[i]
                partner_std[i] = anchor_std[i]

        # 2. Cổng Tương đồng Góc Cosine (Bảo vệ các lớp dị질 như CAFs không bị trộn chéo phân nhóm)
        cos_sim = F.cosine_similarity(anchor_mu, partner_mu, dim=-1, eps=1e-8)
        coherence_gate = (cos_sim >= self.cos_threshold).float().unsqueeze(-1)

        beta_dist = torch.distributions.Beta(self.beta_alpha, self.beta_alpha)
        raw_lam = beta_dist.sample((M, 1)).to(device)
        raw_lam = torch.max(raw_lam, 1.0 - raw_lam)  # Luôn >= 0.5 về phía tế bào neo

        # Nếu 2 tế bào cùng lớp nhưng thuộc 2 phân nhóm quá xa nhau (cos_sim < cos_threshold),
        # thu hẹp hệ số nội suy về [0.90, 0.98] để chỉ mở rộng đám mây phương sai cục bộ của Encoder
        local_lam = 0.90 + 0.08 * torch.rand((M, 1), device=device)
        lam_intra = coherence_gate * raw_lam + (1.0 - coherence_gate) * local_lam

        # Nội suy đồng thời cả kỳ vọng (mu) và độ lệch chuẩn hậu nghiệm (sigma) của Encoder
        mu_mix_intra = lam_intra * anchor_mu + (1.0 - lam_intra) * partner_mu
        std_mix_intra = lam_intra * anchor_std + (1.0 - lam_intra) * partner_std
        eps_intra = torch.randn_like(std_mix_intra)
        z_mix_intra = mu_mix_intra + eps_intra * std_mix_intra

        # 3. Nội suy Biên Bất đối xứng (Asymmetric Boundary Remix) đẩy biên khỏi lớp đa số
        # Tìm điểm khác lớp có độ tương đồng Cosine lớn nhất trong batch (hard boundary negative)
        norm_anchor = F.normalize(anchor_mu.detach(), dim=-1)
        norm_batch = F.normalize(mu.detach(), dim=-1)
        sim_matrix = torch.matmul(norm_anchor, norm_batch.t())  # (M, B)
        same_class_mask = anchor_labels.unsqueeze(1) == labels.unsqueeze(0)
        sim_matrix = sim_matrix.masked_fill(same_class_mask, -1e9)
        hard_neg_idx = sim_matrix.argmax(dim=1)

        neg_mu = mu[hard_neg_idx].detach()
        # Hệ số trộn bám sát tế bào hiếm [0.78, 0.94] để mở rộng lớp vỏ biên mà không xâm lấn vùng Unknown
        lam_bound = 0.78 + 0.16 * torch.rand((M, 1), device=device)
        z_mix_bound = lam_bound * anchor_mu + (1.0 - lam_bound) * neg_mu + 0.5 * torch.randn_like(anchor_std) * anchor_std

        # Gộp cả hai luồng tăng cường (Intra-class + Boundary Remix)
        z_mix_all = torch.cat([z_mix_intra, z_mix_bound], dim=0)
        y_mix_all = torch.cat([anchor_labels, anchor_labels], dim=0)

        # 4. Bộ Chỉnh lưu Đa tạp qua Chu trình Decoder -> Encoder (VAE Manifold Retraction)
        x_surrogate = decoder.decode_to_surrogate_input(z_mix_all)
        _, z_rect_all, _ = encoder(x_surrogate, domain_id=0, freeze_stats=True)

        # Vector tiềm ẩn đã chỉnh lưu kéo về đa tạp sinh học
        z_star = (1.0 - self.rect_blend) * z_mix_all + self.rect_blend * z_rect_all

        # Khoảng cách lệch đa tạp và Trọng số tin cậy sinh học w(z*)
        discrepancy = torch.mean((z_mix_all.detach() - z_rect_all.detach()) ** 2, dim=-1)
        trust_weights = torch.exp(-self.trust_eta * discrepancy)

        # Hàm mất mát Đồng thuận Đa tạp (Manifold Convexity Loss) tính trên các cặp nội lớp hợp lệ
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
# 4. MINIBATCH UNBALANCED OPTIMAL TRANSPORT (MINIBATCH-UOT)
# ==============================================================================
class MinibatchUOT(nn.Module):
    """
    Giải bài toán Vận chuyển Tối ưu Không cân bằng (Unbalanced Optimal Transport)
    giữa các tế bào không gian (Spatial query) và các nguyên mẫu lớp (Prototypes)
    bằng thuật toán Sinkhorn-Knopp khả vi trên miền log.
    """
    def __init__(self, reg_ent: float = 0.05, reg_m_kl: float = 0.5, num_iters: int = 20):
        super().__init__()
        self.reg_ent = reg_ent
        self.reg_m_kl = reg_m_kl
        self.num_iters = num_iters

    @torch.no_grad()
    def compute_transport_plan(
        self, cost_matrix: torch.Tensor, class_prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        cost_matrix: (B_sp, C) khoảng cách Cosine trong [0, 2]
        Trả về kế hoạch vận chuyển T: (B_sp, C)
        """
        B_sp, C = cost_matrix.shape
        device = cost_matrix.device

        # Phân phối biên đều cho các điểm không gian và phân phối biên có điều chỉnh cho nguyên mẫu
        a = torch.full((B_sp,), 1.0 / B_sp, device=device)
        if class_prior is not None:
            b = class_prior / (class_prior.sum() + 1e-8)
        else:
            b = torch.full((C,), 1.0 / C, device=device)

        log_a = torch.log(a + 1e-8)
        log_b = torch.log(b + 1e-8)
        log_K = -cost_matrix / self.reg_ent

        tau = self.reg_m_kl / (self.reg_m_kl + self.reg_ent)
        u = torch.zeros(B_sp, device=device)
        v = torch.zeros(C, device=device)

        for _ in range(self.num_iters):
            u = tau * (log_a - torch.logsumexp(log_K + v.unsqueeze(0), dim=1))
            v = tau * (log_b - torch.logsumexp(log_K + u.unsqueeze(1), dim=0))

        log_T = u.unsqueeze(1) + log_K + v.unsqueeze(0)
        return torch.exp(log_T)


# ==============================================================================
# 5. KIẾN TRÚC TỔNG THỂ: MaRe-VAE (MANIFOLD-RECTIFIED LATENT MIXUP VAE)
# ==============================================================================
class MaReVAE(nn.Module):
    """
    Kiến trúc MaRe-VAE tích hợp:
    - Dual-Domain Variational Encoder (với DSBN tách biệt thống kê)
    - ZINB Generative Decoder
    - Non-GMM, Non-kNN Latent Posterior Mixup & VAE Manifold Rectifier
    - Cosine Prototype Classifier có hiệu chỉnh tần số lớp (Logit Adjustment)
    - Minibatch-UOT & Ràng buộc mượt láng giềng không gian (Spatial Neighbor Smoothness)
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (256, 128),
        class_counts: Optional[torch.Tensor] = None,
        cosine_scale: float = 16.0,
        logit_adj_tau: float = 0.35,
        beta_kl: float = 0.005,
        lambda_cls: float = 2.0,
        lambda_aug: float = 1.0,
        lambda_rect: float = 0.5,
        lambda_uot: float = 0.5,
        lambda_spa: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.cosine_scale = cosine_scale
        self.logit_adj_tau = logit_adj_tau

        # Hệ số các thành phần hàm loss
        self.beta_kl = beta_kl
        self.lambda_cls = lambda_cls
        self.lambda_aug = lambda_aug
        self.lambda_rect = lambda_rect
        self.lambda_uot = lambda_uot
        self.lambda_spa = lambda_spa

        # 1. Encoder & Decoder
        self.encoder = VariationalEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        self.decoder = ZINBDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=(hidden_dims[1], hidden_dims[0]),
        )

        # 2. Trọng số nguyên mẫu phân lớp trên mặt cầu đơn vị (Cosine Prototypes)
        self.prototypes = nn.Parameter(torch.randn(num_classes, latent_dim))
        nn.init.xavier_uniform_(self.prototypes)

        # Tiên nghiệm tần số lớp để điều chỉnh logit nhẹ và cung cấp prior cho UOT
        if class_counts is not None:
            prior = class_counts.float().clamp(min=1.0)
            prior = prior / prior.sum()
        else:
            prior = torch.full((num_classes,), 1.0 / num_classes)
        self.register_buffer("class_prior", prior)
        self.register_buffer("log_class_prior", torch.log(prior + 1e-8))

        # 3. Bộ tăng cường Latent Mixup & Chỉnh lưu Đa tạp (không GMM, không k-NN)
        self.mixup_rectifier = LatentMixupRectifier(
            num_classes=num_classes,
            latent_dim=latent_dim,
            class_counts=class_counts,
        )

        # 4. Module Minibatch-UOT
        self.uot = MinibatchUOT(reg_ent=0.08, reg_m_kl=0.5, num_iters=20)

    def compute_logits(self, z: torch.Tensor, apply_logit_adj: bool = False) -> torch.Tensor:
        """Tính logit phân lớp dựa trên tương đồng Cosine giữa vector tiềm ẩn z và Prototypes."""
        z_norm = F.normalize(z, dim=-1)
        w_norm = F.normalize(self.prototypes, dim=-1)
        logits = self.cosine_scale * torch.matmul(z_norm, w_norm.t())
        if apply_logit_adj and self.logit_adj_tau > 0:
            logits = logits + self.logit_adj_tau * self.log_class_prior.unsqueeze(0)
        return logits

    @staticmethod
    def kl_divergence_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Phân kỳ KL giữa q(z|x) = N(mu, sigma^2) và N(0, I)."""
        return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

    def spatial_smoothness_loss(
        self, probs_spa: torch.Tensor, coords_spa: torch.Tensor, sigma_coord: float = 0.08
    ) -> torch.Tensor:
        """
        Khuyến khích các tế bào nằm gần nhau về mặt tọa độ vật lý có phân phối dự đoán tương đồng
        thông qua hạt nhân Gaussian liên tục trên tọa độ (không cần dựng đồ thị k-NN rời rạc).
        """
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
        """
        Thực hiện 1 bước huấn luyện tích hợp trên cặp mini-batch scRNA-seq và Spatial.
        - Khi ở giai đoạn Warm-up: đặt `use_latent_mixup=False` để hình thành đa tạp gốc.
        - Khi vào giai đoạn chính: đặt `use_latent_mixup=True` để kích hoạt Latent Mixup + Rectifier.
        """
        x_rna = batch_rna["x"]
        raw_rna = batch_rna["raw_counts"]
        lib_rna = batch_rna["library_size"]
        y_rna = batch_rna["label"]

        # 1. Nhánh scRNA-seq thực (Origin manifold)
        z_rna, mu_rna, logvar_rna = self.encoder(x_rna, domain_id=0, freeze_stats=False)
        rho_rna, pi_rna, theta = self.decoder(z_rna)

        recon_loss_rna = zinb_negative_log_likelihood(raw_rna, lib_rna, rho_rna, pi_rna, theta)
        kl_loss_rna = self.kl_divergence_normal(mu_rna, logvar_rna)

        logits_rna = self.compute_logits(z_rna, apply_logit_adj=True)
        cls_loss_real = F.cross_entropy(logits_rna, y_rna)

        total_loss = recon_loss_rna + self.beta_kl * kl_loss_rna + self.lambda_cls * cls_loss_real

        # 2. Nhánh Latent Mixup & Chỉnh lưu Đa tạp (nếu được kích hoạt)
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

            logits_aug = self.compute_logits(z_aug, apply_logit_adj=False)
            per_sample_ce = F.cross_entropy(logits_aug, y_aug, reduction="none")
            aug_cls_loss = torch.mean(trust_w * per_sample_ce)

            total_loss = total_loss + self.lambda_aug * aug_cls_loss + self.lambda_rect * rect_loss
        else:
            # Trong pha warm-up vẫn cập nhật hàng đợi để sẵn sàng cho pha sau
            self.mixup_rectifier.update_queues(mu_rna, logvar_rna, y_rna)

        # 3. Nhánh Spatial Transcriptomics & Căn chỉnh Minibatch-UOT
        recon_loss_spa = torch.tensor(0.0, device=x_rna.device)
        uot_loss = torch.tensor(0.0, device=x_rna.device)
        spa_smooth_loss = torch.tensor(0.0, device=x_rna.device)

        if batch_spa is not None:
            x_spa = batch_spa["x"]
            raw_spa = batch_spa["raw_counts"]
            lib_spa = batch_spa["library_size"]
            coords_spa = batch_spa["coords"]

            z_spa, mu_spa, logvar_spa = self.encoder(x_spa, domain_id=1, freeze_stats=False)
            rho_spa, pi_spa, theta_spa = self.decoder(z_spa)

            recon_loss_spa = zinb_negative_log_likelihood(raw_spa, lib_spa, rho_spa, pi_spa, theta_spa)
            kl_loss_spa = self.kl_divergence_normal(mu_spa, logvar_spa)

            # Tính khoảng cách Cosine từ điểm không gian tới các nguyên mẫu lớp
            z_spa_norm = F.normalize(mu_spa, dim=-1)
            w_norm = F.normalize(self.prototypes, dim=-1)
            cos_sim_spa = torch.matmul(z_spa_norm, w_norm.t())
            cost_matrix = 1.0 - cos_sim_spa  # (B_sp, C) trong [0, 2]

            # Kế hoạch vận chuyển tối ưu không cân bằng (UOT) với tiên nghiệm lớp đã làm mượt
            tempered_prior = self.class_prior.pow(0.5)
            transport_plan = self.uot.compute_transport_plan(cost_matrix.detach(), class_prior=tempered_prior)
            pseudo_probs = transport_plan / (transport_plan.sum(dim=1, keepdim=True) + 1e-8)

            # Trọng số tin cậy của kế hoạch vận chuyển cho từng tế bào không gian
            cell_mass = transport_plan.sum(dim=1)
            mass_weights = torch.clamp(cell_mass * cost_matrix.shape[0], min=0.1, max=2.0).detach()

            logits_spa = self.compute_logits(mu_spa, apply_logit_adj=False)
            log_probs_spa = F.log_softmax(logits_spa, dim=-1)
            uot_ce = -torch.sum(pseudo_probs * log_probs_spa, dim=-1)
            uot_loss = torch.mean(mass_weights * uot_ce)

            probs_spa = F.softmax(logits_spa, dim=-1)
            spa_smooth_loss = self.spatial_smoothness_loss(probs_spa, coords_spa)

            total_loss = (
                total_loss
                + recon_loss_spa
                + self.beta_kl * kl_loss_spa
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
        unknown_quantile: float = 0.25,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Dự đoán nhãn, xác suất phân lớp, vector tiềm ẩn và điểm năng lượng (cho bài toán Open-set)
        trên toàn bộ một DataLoader.
        - Nếu `detect_unknown=True` (dùng cho tập seqFISH có lớp Unknown): các tế bào có độ tin cậy
          năng lượng tự do + khối lượng UOT thấp dưới ngưỡng sẽ được gán nhãn `-1` (Unknown).
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
        tempered_prior = self.class_prior.pow(0.5)

        for batch in dataloader:
            x = batch["x"].to(device)
            labels = batch["label"]
            _, mu, _ = self.encoder(x, domain_id=domain_id, freeze_stats=True)

            logits = self.compute_logits(mu, apply_logit_adj=False)
            probs = F.softmax(logits, dim=-1)

            # Điểm tin cậy kết hợp giữa Free Energy, Max Softmax và Khối lượng vận chuyển UOT
            max_prob, _ = probs.max(dim=-1)
            free_energy_conf = torch.logsumexp(logits, dim=-1) / self.cosine_scale

            cost_matrix = 1.0 - torch.matmul(F.normalize(mu, dim=-1), w_norm.t())
            T_plan = self.uot.compute_transport_plan(cost_matrix, class_prior=tempered_prior)
            uot_mass = T_plan.sum(dim=-1) * cost_matrix.shape[0]

            known_score = 0.4 * max_prob + 0.4 * free_energy_conf + 0.2 * uot_mass

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
