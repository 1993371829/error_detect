"""
Stage 2 模型：去噪自编码器 DAE 与 GANomaly，二者角色分工。

角色分工（见 stage_2/DESIGN.md）:
    - DAE：单元格级定位专家。混合输出头（数值线性 / 类别 softmax / 缺失·UNK sigmoid）、
      按"原始列"遮蔽的去噪训练（含可选 swap-noise）、按列损失（类别 CE + 数值 MSE）。
      重构误差天然可还原到列，适合类别/缺失/拼写错误的单元格定位。
    - GANomaly：行级 / 多列联合异常专家。混合生成头 + feature-matching + 稳定化
      （spectral norm / label smoothing / 权重重平衡）。异常分用"隐空间偏差 +
      判别器特征偏差"的鲁棒组合，适合发现整行级别的联合异常；逐列贡献用于粗定位。

公共接口:
    fit(x_clean, feature_spec=None)   在干净特征矩阵上训练
    reconstruct(x)                    返回重构矩阵（已施加混合头激活）
    per_feature_error(x)              返回 |x - reconstruct| 的逐特征误差
    anomaly_score(x)                  返回逐行异常分

PyTorch 为可选依赖；CPU 训练即可（本数据集千行级）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _require_torch():
    """惰性导入 torch，缺失时给出清晰报错。"""
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Stage 2 模型训练需要 PyTorch。请先安装: "
            'pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu'
        ) from exc


def _seed_everything(seed: int) -> None:
    """固定随机种子，保证可复现。"""
    torch = _require_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)


class _Heads:
    """把 FeatureSpec 转成 torch 张量索引，并提供混合头激活、混合损失、按列遮蔽。"""

    def __init__(self, d: int, feature_spec, device: str):
        torch = _require_torch()
        self.d = d
        self.device = device
        if feature_spec is None:
            # 无 spec：退化为全数值（纯 MSE 自编码器）
            self.numeric_idx = torch.arange(d, device=device)
            self.binary_idx = torch.empty(0, dtype=torch.long, device=device)
            self.softmax_groups: list[tuple[int, int]] = []
            self.column_slices: list[tuple[int, int]] = [(0, d)]
        else:
            self.numeric_idx = torch.tensor(feature_spec.numeric_idx, dtype=torch.long, device=device)
            self.binary_idx = torch.tensor(feature_spec.binary_idx, dtype=torch.long, device=device)
            self.softmax_groups = list(feature_spec.softmax_groups)
            self.column_slices = list(feature_spec.column_slices.values())
        self._n_groups = max(1, len(self.softmax_groups))

    def activate(self, raw):
        """对网络原始输出施加混合头：数值线性、binary sigmoid、softmax 段归一。"""
        torch = _require_torch()
        out = raw.clone()
        if self.binary_idx.numel():
            out[:, self.binary_idx] = torch.sigmoid(raw[:, self.binary_idx])
        for s, e in self.softmax_groups:
            out[:, s:e] = torch.softmax(raw[:, s:e], dim=1)
        return out

    def loss(self, raw, target):
        """混合重构损失：数值 MSE + binary BCE(logits) + 类别段 CE（按组平均）。"""
        torch = _require_torch()
        import torch.nn.functional as F

        total = raw.new_zeros(())
        if self.numeric_idx.numel():
            total = total + F.mse_loss(raw[:, self.numeric_idx], target[:, self.numeric_idx])
        if self.binary_idx.numel():
            total = total + F.binary_cross_entropy_with_logits(
                raw[:, self.binary_idx], target[:, self.binary_idx]
            )
        if self.softmax_groups:
            ce = raw.new_zeros(())
            for s, e in self.softmax_groups:
                tgt_cls = target[:, s:e].argmax(dim=1)
                ce = ce + F.cross_entropy(raw[:, s:e], tgt_cls)
            total = total + ce / self._n_groups
        return total

    def corrupt(self, x, rate: float, swap_rate: float):
        """按原始列遮蔽：每行每列以 rate 概率被破坏（swap_rate 概率与他行交换，否则置零）。"""
        torch = _require_torch()
        n = x.shape[0]
        corrupted = x.clone()
        for s, e in self.column_slices:
            col_mask = torch.rand(n, device=self.device) < rate
            if not bool(col_mask.any()):
                continue
            do_swap = col_mask & (torch.rand(n, device=self.device) < swap_rate)
            do_zero = col_mask & ~do_swap
            if bool(do_zero.any()):
                corrupted[do_zero, s:e] = 0.0
            if bool(do_swap.any()):
                k = int(do_swap.sum())
                src = torch.randint(0, n, (k,), device=self.device)
                corrupted[do_swap, s:e] = x[src, s:e]
        return corrupted


class BaseAnomalyModel:
    """异常检测模型统一接口。"""

    def fit(self, x_clean: np.ndarray, feature_spec=None) -> "BaseAnomalyModel":
        raise NotImplementedError

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def per_feature_error(self, x: np.ndarray) -> np.ndarray:
        """逐特征绝对重构误差，形状同 x。"""
        recon = self.reconstruct(x)
        return np.abs(x - recon)

    def anomaly_score(self, x: np.ndarray) -> np.ndarray:
        """逐行异常分（默认用逐特征误差求和）。"""
        return self.per_feature_error(x).sum(axis=1)


class DenoisingAutoencoder(BaseAnomalyModel):
    """
    去噪自编码器：单元格级定位专家。

    - 混合输出头：数值线性、类别 one-hot softmax、缺失/UNK sigmoid。
    - 去噪：按"原始列"整列遮蔽（置零或 swap-noise），强迫用其他列上下文还原。
    - 损失：类别段交叉熵 + 数值 MSE + 指示位 BCE，缓解稀疏 one-hot 的 MSE 偏置。

    Args:
        hidden_dim / latent_dim: 网络维度。
        corruption_rate: 训练时每行被遮蔽的"列"比例。
        swap_rate: 被遮蔽列中改用 swap-noise（他行同列值替换）的比例，否则置零。
        epochs / lr / batch_size: 训练超参。
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        corruption_rate: float = 0.2,
        swap_rate: float = 0.5,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 128,
        seed: int = 0,
        verbose: bool = True,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.corruption_rate = corruption_rate
        self.swap_rate = swap_rate
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self._net = None
        self._heads: Optional[_Heads] = None
        self._device = "cpu"

    def _build_net(self, d: int):
        """对称 Encoder-Decoder（瓶颈在 latent_dim），输出为原始 logits。"""
        _require_torch()
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(d, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, d),
        )

    def fit(self, x_clean: np.ndarray, feature_spec=None) -> "DenoisingAutoencoder":
        torch = _require_torch()
        _seed_everything(self.seed)

        d = x_clean.shape[1]
        self._net = self._build_net(d).to(self._device)
        self._heads = _Heads(d, feature_spec, self._device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        x = torch.tensor(np.asarray(x_clean, dtype=np.float32), device=self._device)
        n = x.shape[0]

        self._net.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self._device)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                target = x[idx]
                corrupted = self._heads.corrupt(target, self.corruption_rate, self.swap_rate)
                raw = self._net(corrupted)
                loss = self._heads.loss(raw, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                print(f"  [DAE] epoch {epoch + 1}/{self.epochs} loss={epoch_loss / n:.5f}")
        return self

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("模型未训练，先调用 fit()。")
        torch = _require_torch()
        self._net.eval()
        with torch.no_grad():
            xt = torch.tensor(np.asarray(x, dtype=np.float32), device=self._device)
            recon = self._heads.activate(self._net(xt))
        return recon.cpu().numpy()


class GANomaly(BaseAnomalyModel):
    """
    GANomaly：行级 / 多列联合异常专家。

    结构: Encoder1 -> Decoder(G) -> Encoder2，外加判别器 D（返回 logits 与中间特征）。
    损失:
        - 混合重构  mixed(G(E1(x)), x)
        - 隐一致    ||E1(x) - E2(G(x))||
        - 对抗      D 区分真实 x 与重构（label smoothing）
        - feature-matching  ||mean f(x) - mean f(G(x))||（稳定 GAN）
    异常分: 隐空间偏差与判别器特征偏差的鲁棒标准化之和（行级）。
    逐列贡献: |x - G(x)| 还原到列（粗定位）。

    Args:
        latent_dim / hidden_dim: 网络维度。
        w_recon / w_latent / w_adv / w_fm: 各损失项权重。
        epochs / lr / batch_size: 训练超参。
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        w_recon: float = 10.0,
        w_latent: float = 1.0,
        w_adv: float = 1.0,
        w_fm: float = 1.0,
        epochs: int = 200,
        lr: float = 2e-4,
        batch_size: int = 128,
        label_smooth: float = 0.9,
        seed: int = 0,
        verbose: bool = True,
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.w_recon = w_recon
        self.w_latent = w_latent
        self.w_adv = w_adv
        self.w_fm = w_fm
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.label_smooth = label_smooth
        self.seed = seed
        self.verbose = verbose
        self._e1 = self._g = self._e2 = self._d = None
        self._heads: Optional[_Heads] = None
        self._device = "cpu"

    def _build_blocks(self, d: int):
        """构建 E1 / G / E2 / D；D 用 spectral norm 并返回中间特征。"""
        _require_torch()
        import torch.nn as nn
        from torch.nn.utils import spectral_norm

        e1 = nn.Sequential(
            nn.Linear(d, self.hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        g = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, d),
        )
        e2 = nn.Sequential(
            nn.Linear(d, self.hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

        class _Disc(nn.Module):
            def __init__(self, d_in, hidden):
                super().__init__()
                self.body = nn.Sequential(
                    spectral_norm(nn.Linear(d_in, hidden)), nn.LeakyReLU(0.2),
                    spectral_norm(nn.Linear(hidden, hidden)), nn.LeakyReLU(0.2),
                )
                self.head = spectral_norm(nn.Linear(hidden, 1))

            def forward(self, x):
                feat = self.body(x)
                return self.head(feat), feat

        return e1, g, e2, _Disc(d, self.hidden_dim)

    def fit(self, x_clean: np.ndarray, feature_spec=None) -> "GANomaly":
        torch = _require_torch()
        _seed_everything(self.seed)
        import itertools

        d = x_clean.shape[1]
        self._e1, self._g, self._e2, self._d = (
            m.to(self._device) for m in self._build_blocks(d)
        )
        self._heads = _Heads(d, feature_spec, self._device)
        gen_params = itertools.chain(
            self._e1.parameters(), self._g.parameters(), self._e2.parameters()
        )
        opt_g = torch.optim.Adam(gen_params, lr=self.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self._d.parameters(), lr=self.lr, betas=(0.5, 0.999))
        l_lat = torch.nn.MSELoss()
        l_adv = torch.nn.BCEWithLogitsLoss()

        x = torch.tensor(np.asarray(x_clean, dtype=np.float32), device=self._device)
        n = x.shape[0]

        for net in (self._e1, self._g, self._e2, self._d):
            net.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self._device)
            g_acc = d_acc = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                real = x[idx]
                bs = real.shape[0]
                real_lbl = torch.full((bs, 1), self.label_smooth, device=self._device)
                fake_lbl = torch.zeros(bs, 1, device=self._device)
                ones = torch.ones(bs, 1, device=self._device)

                z = self._e1(real)
                x_hat = self._heads.activate(self._g(z))

                # --- 判别器 ---
                opt_d.zero_grad()
                d_real, _ = self._d(real)
                d_fake, _ = self._d(x_hat.detach())
                loss_d = l_adv(d_real, real_lbl) + l_adv(d_fake, fake_lbl)
                loss_d.backward()
                opt_d.step()

                # --- 生成器：混合重构 + 隐一致 + 对抗 + feature-matching ---
                opt_g.zero_grad()
                z_hat = self._e2(x_hat)
                d_fake2, f_fake = self._d(x_hat)
                _, f_real = self._d(real)
                loss_g = (
                    self.w_recon * self._heads.loss(self._g(z), real)
                    + self.w_latent * l_lat(z_hat, z)
                    + self.w_adv * l_adv(d_fake2, ones)
                    + self.w_fm * l_lat(f_fake.mean(0), f_real.detach().mean(0))
                )
                loss_g.backward()
                opt_g.step()

                g_acc += loss_g.item() * bs
                d_acc += loss_d.item() * bs
            if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                print(f"  [GANomaly] epoch {epoch + 1}/{self.epochs} "
                      f"g={g_acc / n:.4f} d={d_acc / n:.4f}")
        return self

    def _forward_eval(self, x: np.ndarray):
        torch = _require_torch()
        for net in (self._e1, self._g, self._e2, self._d):
            net.eval()
        with torch.no_grad():
            xt = torch.tensor(np.asarray(x, dtype=np.float32), device=self._device)
            z = self._e1(xt)
            x_hat = self._heads.activate(self._g(z))
            z_hat = self._e2(x_hat)
            _, f_real = self._d(xt)
            _, f_fake = self._d(x_hat)
        return xt, x_hat, z, z_hat, f_real, f_fake

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        if self._g is None:
            raise RuntimeError("模型未训练，先调用 fit()。")
        _, x_hat, *_ = self._forward_eval(x)
        return x_hat.cpu().numpy()

    def anomaly_score(self, x: np.ndarray) -> np.ndarray:
        """隐空间偏差 + 判别器特征偏差（各自鲁棒标准化后相加）的行级异常分。"""
        if self._g is None:
            raise RuntimeError("模型未训练，先调用 fit()。")
        torch = _require_torch()
        _, _, z, z_hat, f_real, f_fake = self._forward_eval(x)
        latent = torch.norm(z - z_hat, dim=1).cpu().numpy()
        feat = torch.norm(f_real - f_fake, dim=1).cpu().numpy()
        return _robust_z(latent) + _robust_z(feat)


def _robust_z(v: np.ndarray) -> np.ndarray:
    """中位数/MAD 鲁棒标准化。"""
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) or 1.0
    return (v - med) / (1.4826 * mad)


def build_model(name: str = "dae", **kwargs) -> BaseAnomalyModel:
    """工厂：按名称构建模型（'dae' 单元格级 / 'ganomaly' 行级）。"""
    name = name.lower()
    if name in ("ganomaly", "gan"):
        return GANomaly(**kwargs)
    if name in ("dae", "autoencoder", "ae"):
        return DenoisingAutoencoder(**kwargs)
    raise ValueError(f"未知模型: {name!r}，可选 'ganomaly' 或 'dae'。")
