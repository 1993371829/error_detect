"""
Stage 2 包：无监督异常检测层（Layer 2），统一条件预测模型。

承接 Stage 1 的规则层输出，在"干净子集"上学习每一列的条件分布
P(列 | 其余列)。一个单元格的取值若难以由同一行其余列预测出来，则可疑。
该单一机制同时覆盖:
    - 分布型异常（拼写 / 未知类别 / 数值越界）：行上下文给出低条件概率。
    - 键 -> 值冲突 / 真值发现（如某来源把航班时刻写错）：P(time|flight)
      集中在该航班的共识取值，偏离者得到低概率。

模块概览:
    encoding.py  : 表格 -> 张量编码（identity-preserving 输入 + 逐列目标头规格）
    model.py     : ConditionalPredictor（masked-column 训练的共享编码器 + 逐列头）
    score.py     : 逐格似然 / 残差 + 干净分位阈值 + 精度闸门 + suggested_fix

详见 stage_2/DESIGN.md。
"""

__all__ = []
