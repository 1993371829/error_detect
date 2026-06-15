"""
Stage 2 包：GAN 分布异常检测层（Layer 2）。

承接 Stage 1 的规则层输出，在"干净子集"上学习表格的联合分布，
对每一行给出异常分，并把异常分摊到各列（逐列重构误差贡献），
用于捕获规则层抓不到的:
    - 跨列依赖违反（VAD，如 city-state-zip 不匹配）
    - 数值离群 / 分布异常

主线模型: 重构式 GAN（GANomaly 风格）。
对照基线: 去噪自编码器（DAE）。

模块概览:
    encoding.py  : 表格 -> 张量编码（数值归一化 + 类别编码）
    model.py     : GANomaly / DAE 骨架
    score.py     : 行级异常分 + 逐列贡献 + 无监督阈值

详见 stage_2/DESIGN.md。
"""

__all__ = []
