"""
Stage 2 模型：统一的条件预测网络 ConditionalPredictor。

核心思想（见 stage_2/DESIGN.md）:
    一个单元格是否为错误，取决于"它的取值能否由同一行的其余列预测出来"。
    模型学习每一列的条件分布 P(列 | 其余列)：
        - 类别列：softmax 给出各类别概率；观测值的负对数似然(NLL)越大越可疑。
        - 数值列：回归预测，标准化残差越大越可疑。
        - 超高基数文本列：重构 6 个形态特征（兜底），MSE 越大越可疑。

    该机制同时统一了两类错误：
        - 分布型异常（hospital：拼写/未知类别/越界）-> 行上下文给出低概率。
        - 键->值冲突 / 真值发现（flights：某来源把航班时刻写错）-> P(time|flight)
          会集中在该航班的共识取值上，偏离的时刻自然得到低概率。

训练（仅用整行干净样本，masked-column modeling）:
    每个 batch 对每一列 c：把 c 的输入块置零（屏蔽自身），用其余列预测 c 的目标，
    累加各列损失（类别 CE / 数值 MSE / surrogate MSE）。
推理:
    对每一列 c 同样屏蔽自身后预测，得到该列在每一行上的条件分布 / 预测值，
    交由 score.py 计算逐格似然/残差并按干净分位阈值筛选。

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(pref: str) -> str:
    """
    解析设备偏好为实际设备字符串。
        - "auto": 有 CUDA 用 "cuda"，否则 "cpu"（服务器零配置自动启用 GPU）。
        - "cuda"/"cuda:0"/...: 显式使用 GPU；若环境无 CUDA 则回退 "cpu" 并告警。
        - "cpu": 强制 CPU。
    """
    torch = _require_torch()
    pref = (pref or "auto").lower()
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref.startswith("cuda"):
        if torch.cuda.is_available():
            return pref
        print(f"  [CondPred] 警告: 指定 device={pref} 但未检测到 CUDA，回退 CPU。")
        return "cpu"
    return "cpu"


class ConditionalPredictor:
    """
    统一条件预测模型：共享编码器 + 逐列预测头，masked-column 训练。

    公共接口:
        fit(x_clean, specs)   在干净特征矩阵上训练（specs 来自 encoder.column_specs()）
        predict(x)            返回 {col_name: 预测物}，其中
                                  categorical -> softmax 概率 (n, onehot_dim)
                                  numeric     -> 预测标量 (n,)
                                  surrogate   -> 重构特征 (n, surrogate_dim)

    Args:
        hidden_dim: 编码器隐藏维。
        epochs / lr / batch_size: 训练超参。
        weight_decay: L2 正则（缓解键->值的过拟合记忆带来的过度自信）。
        seed / verbose: 复现与日志。
        device: 计算设备偏好，"auto"（默认，有 GPU 自动用）/ "cuda" / "cpu"。
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 256,
        weight_decay: float = 1e-5,
        seed: int = 0,
        verbose: bool = True,
        device: str = "auto",
        val_split: float = 0.15,
        patience: int = 15,
        min_delta: float = 1e-4,
    ):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = seed
        self.verbose = verbose
        self.device_pref = device
        # 早停：留出验证集，val_loss 连续 patience 轮无改善即停并回滚到最优权重。
        # val_split<=0 或样本过少时退回「训练全量 + 跑满 epochs」的旧行为。
        self.val_split = val_split
        self.patience = patience
        self.min_delta = min_delta
        self._encoder = None
        self._heads = None          # torch ModuleDict，键为 "h{idx}"
        self._specs = None
        self._head_key: dict[str, str] = {}   # col name -> head key
        self._spec_col: dict[str, int] = {}   # col name -> specs 列表下标（cell_valid 对齐用）
        self._device = "cpu"        # 实际设备，fit() 时按 device_pref 解析

    # ------------------------------------------------------------------ build
    def _build(self, d: int):
        _require_torch()
        import torch.nn as nn

        encoder = nn.Sequential(
            nn.Linear(d, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
        )
        heads = nn.ModuleDict()
        self._head_key = {}
        for i, spec in enumerate(self._specs):
            out_dim = self._target_dim(spec)
            if out_dim > 0:
                key = f"h{i}"
                heads[key] = nn.Linear(self.hidden_dim, out_dim)
                self._head_key[spec.name] = key
        return encoder, heads

    @staticmethod
    def _target_dim(spec) -> int:
        if spec.target_kind == "categorical":
            return spec.onehot_dim
        if spec.target_kind == "numeric":
            return 1
        if spec.target_kind == "surrogate":
            return spec.surrogate_dim
        return 0

    # ------------------------------------------------------------------ targets
    @staticmethod
    def _targets_for(spec, x):
        """从（未屏蔽的）输入张量中取出该列的训练目标与有效掩码。"""
        torch = _require_torch()
        s = spec.start
        if spec.target_kind == "categorical":
            onehot = x[:, s:s + spec.onehot_dim]
            target = onehot.argmax(dim=1)                 # 类别 id（含 UNK）
            valid = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            return target, valid
        if spec.target_kind == "numeric":
            target = x[:, spec.value_index]
            valid = x[:, spec.isnull_index] < 0.5         # 非空才参与
            return target, valid
        if spec.target_kind == "surrogate":
            target = x[:, s:s + spec.surrogate_dim]
            valid = x[:, spec.isnull_index] < 0.5
            return target, valid
        return None, None

    def _masked_forward(self, x, spec):
        """屏蔽 spec 列的输入块后过编码器，返回该列头输出。"""
        torch = _require_torch()
        xm = x.clone()
        xm[:, spec.start:spec.start + spec.width] = 0.0
        h = self._encoder(xm)
        return self._heads[self._head_key[spec.name]](h)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        x_clean: np.ndarray,
        specs: list,
        cell_valid: Optional[np.ndarray] = None,
        x_input: Optional[np.ndarray] = None,
    ) -> "ConditionalPredictor":
        """
        在（伪）干净特征矩阵上训练。

        Args:
            x_clean:  取训练目标（真值）的特征矩阵 (n, d)。
            specs:    列规格列表（来自 encoder.column_specs()）。
            cell_valid: 可选 cell 级有效掩码 (n, len(specs))，按 specs 顺序对齐；
                        False 表示该 (行, 列) 被屏蔽（伪干净构造里的单错格），
                        其目标不计入 loss。None 时全部有效（旧行为）。
            x_input:  可选上下文输入矩阵 (n, d)，用于「作为其余列的上下文」；被屏蔽格
                        在此已中性化（见 score.neutralize_context）。None 时等于 x_clean。
        """
        torch = _require_torch()
        import torch.nn as nn
        import torch.nn.functional as F

        _seed_everything(self.seed)
        self._device = _resolve_device(self.device_pref)
        if self.verbose:
            print(f"  [CondPred] device={self._device}")
        self._specs = list(specs)
        self._spec_col = {s.name: i for i, s in enumerate(self._specs)}
        d = x_clean.shape[1]
        self._encoder, self._heads = self._build(d)
        self._encoder.to(self._device)
        self._heads.to(self._device)

        params = list(self._encoder.parameters()) + list(self._heads.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        x = torch.tensor(np.asarray(x_clean, dtype=np.float32), device=self._device)
        # 上下文输入：默认与目标一致；伪干净时传入中性化后的矩阵以隔离被屏蔽格
        if x_input is not None:
            x_in = torch.tensor(np.asarray(x_input, dtype=np.float32), device=self._device)
        else:
            x_in = x
        cv = None
        if cell_valid is not None:
            cv = torch.tensor(np.asarray(cell_valid, dtype=bool), device=self._device)
        n = x.shape[0]
        trainable = [s for s in self._specs if self._target_dim(s) > 0]

        # 划分训练/验证子集（样本足够时才启用早停，否则全量训练跑满 epochs）
        gen = torch.Generator(device=self._device).manual_seed(self.seed)
        perm0 = torch.randperm(n, generator=gen, device=self._device)
        use_val = self.val_split and self.val_split > 0 and n >= 20
        if use_val:
            n_val = max(1, int(round(n * self.val_split)))
            val_idx = perm0[:n_val]
            train_idx = perm0[n_val:]
        else:
            val_idx = None
            train_idx = perm0
        n_train = int(train_idx.shape[0])

        best_val = float("inf")
        best_state = None
        no_improve = 0

        self._encoder.train()
        self._heads.train()
        for epoch in range(self.epochs):
            self._encoder.train()
            self._heads.train()
            perm = train_idx[torch.randperm(n_train, device=self._device)]
            epoch_loss = 0.0
            for start in range(0, n_train, self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch = x[idx]              # 目标（真值）来源
                batch_in = x_in[idx]        # 上下文输入来源（被屏蔽格已中性化）
                cv_batch = cv[idx] if cv is not None else None
                optimizer.zero_grad()
                loss = self._batch_loss(batch, batch_in, cv_batch, trainable, F)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(idx)

            if not use_val:
                if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                    print(f"  [CondPred] epoch {epoch + 1}/{self.epochs} loss={epoch_loss / n_train:.5f}")
                continue

            # 验证集损失 + 早停
            val_loss = self._eval_loss(x, x_in, cv, val_idx, trainable, F)
            if val_loss < best_val - self.min_delta:
                best_val = val_loss
                best_state = self._state_snapshot()
                no_improve = 0
            else:
                no_improve += 1
            if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                print(f"  [CondPred] epoch {epoch + 1}/{self.epochs} "
                      f"train={epoch_loss / n_train:.5f} val={val_loss:.5f} best={best_val:.5f}")
            if no_improve >= self.patience:
                if self.verbose:
                    print(f"  [CondPred] 早停于 epoch {epoch + 1}（val 连续 {self.patience} 轮无改善）")
                break

        if use_val and best_state is not None:
            self._load_snapshot(best_state)
        return self

    # ------------------------------------------------------------------ loss helpers
    def _batch_loss(self, batch, batch_in, cv_batch, trainable, F):
        """一个 batch 的多列联合 loss（训练/验证共用）。"""
        loss = batch.new_zeros(())
        for spec in trainable:
            out = self._masked_forward(batch_in, spec)
            target, valid = self._targets_for(spec, batch)
            # cell 级掩码：被屏蔽格（伪干净的单错格）不计入该列 loss
            if cv_batch is not None:
                valid = valid & cv_batch[:, self._spec_col[spec.name]]
            if spec.target_kind == "categorical":
                if bool(valid.all()):
                    loss = loss + F.cross_entropy(out, target)
                elif bool(valid.any()):
                    loss = loss + F.cross_entropy(out[valid], target[valid])
            elif spec.target_kind == "numeric":
                if bool(valid.any()):
                    loss = loss + F.mse_loss(out[:, 0][valid], target[valid])
            elif spec.target_kind == "surrogate":
                if bool(valid.any()):
                    loss = loss + F.mse_loss(out[valid], target[valid])
        return loss

    def _eval_loss(self, x, x_in, cv, val_idx, trainable, F) -> float:
        """验证集平均 batch loss（no_grad）。"""
        torch = _require_torch()
        self._encoder.eval()
        self._heads.eval()
        n_val = int(val_idx.shape[0])
        total, n_batches = 0.0, 0
        with torch.no_grad():
            for start in range(0, n_val, self.batch_size):
                idx = val_idx[start:start + self.batch_size]
                cv_batch = cv[idx] if cv is not None else None
                loss = self._batch_loss(x[idx], x_in[idx], cv_batch, trainable, F)
                total += float(loss.item())
                n_batches += 1
        return total / max(1, n_batches)

    def _state_snapshot(self) -> dict:
        """深拷贝当前编码器/头权重（供早停回滚）。"""
        import copy
        return {
            "encoder": copy.deepcopy(self._encoder.state_dict()),
            "heads": copy.deepcopy(self._heads.state_dict()),
        }

    def _load_snapshot(self, state: dict) -> None:
        self._encoder.load_state_dict(state["encoder"])
        self._heads.load_state_dict(state["heads"])

    # ------------------------------------------------------------------ predict
    def predict(self, x: np.ndarray) -> dict:
        """对每列屏蔽自身后预测，返回 {col: 概率/标量/重构}。"""
        if self._encoder is None:
            raise RuntimeError("模型未训练，先调用 fit()。")
        torch = _require_torch()
        self._encoder.eval()
        self._heads.eval()
        out: dict[str, np.ndarray] = {}
        with torch.no_grad():
            xt = torch.tensor(np.asarray(x, dtype=np.float32), device=self._device)
            for spec in self._specs:
                if self._target_dim(spec) == 0:
                    continue
                raw = self._masked_forward(xt, spec)
                if spec.target_kind == "categorical":
                    out[spec.name] = torch.softmax(raw, dim=1).cpu().numpy()
                elif spec.target_kind == "numeric":
                    out[spec.name] = raw[:, 0].cpu().numpy()
                elif spec.target_kind == "surrogate":
                    out[spec.name] = raw.cpu().numpy()
        return out


def build_model(**kwargs) -> ConditionalPredictor:
    """工厂：构建统一条件预测模型。"""
    return ConditionalPredictor(**kwargs)
