# 本地验证记录

验证日期：2026-09-14。所有实现均位于refactored_framework，原stage_1/2/3未修改。未调用付费LLM，未下载BGE权重。

## 已通过

| 检查 | 结果 |
|---|---|
| Python源码编译 | 通过 |
| pytest完整套件 | **26 passed**，26.92秒 |
| PyTorch CPU前向、反向、参数更新 | 通过，使用真实模型计算 |
| 早停、检查点保存/恢复和预测重放 | 通过 |
| 三折训练/评分行隔离、重复行同折 | 通过 |
| 修改clean文件不影响训练和检测 | 通过 |
| 有效结构诱导、伪标签和第二轮真实训练 | 通过，外部API使用测试响应 |
| 非法结构、矛盾伪标签、缺失计数器、预算耗尽 | 通过 |
| 失败重试与未知费用预留、恢复账本、缓存 | 通过 |
| Duration/ratio、单位不匹配、受限表达式、正则超时 | 通过 |
| FD左侧/右侧定位、多字段差异保留未决 | 通过 |
| 独立目录复制运行、禁止旧框架导入 | 通过 |
| wheel构建与隔离目标目录安装 | 通过 |
| 安装后的doctor及完整样例run | 通过 |
| Bash及PowerShell脚本语法 | 通过 |

测试命令：

```bash
python -B -m pytest -q -o faulthandler_timeout=90 --basetemp outputs/pytest_release --junitxml=outputs/pytest_release_results.xml
python -m pip wheel . --no-deps --no-build-isolation --no-cache-dir --wheel-dir dist
```

26个测试覆盖19个单元/模型场景和7个集成场景。测试期间禁止网络连接；独立子进程显式使用fixture模式。详细JUnit和运行输出留在本地outputs，上传包不携带这些缓存。

## 六表格式兼容性冒烟

每表取原脏表前32行，seed=42、三折、两轮真实CPU训练、词法编码、固定LLM响应。不是完整数据集质量实验。以下“计量单位”是fixture UTF-8字节代理，不是真实API tokenizer token，也不代表账单。

| 数据集 | 形状 | 耗时/秒 | 模拟请求 | 计量单位 |
|---|---:|---:|---:|---:|
| hospital | 32×17 | 17.41 | 4 | 57,887 |
| flights | 32×6 | 5.44 | 6 | 35,632 |
| beers | 32×9 | 6.47 | 7 | 50,078 |
| rayyan | 32×11 | 5.33 | 4 | 55,791 |
| movies | 32×17 | 7.52 | 2 | 41,417 |
| billionaire | 32×22 | 10.56 | 4 | 66,504 |

全部完成并输出每个单元格的判定。部分阶段因较保守的字节计数触及阶段预算而转为本地执行，report.json会记录budget exhausted；这是预期的预算约束行为，不是隐瞒失败。该轮之后增加的正则执行超时保护已由最终完整测试覆盖。

## 环境和遇到的问题

Windows、本地Python 3.9.13、PyTorch 2.8.0+cpu、NumPy 1.26.4、pandas 2.3.3、scikit-learn 1.6.1、pytest 7.1.2、Pydantic 2.12.5。

- 已修复Python3.9中Pydantic不能直接解析`bool | None`字段的问题，改用Optional。
- 旧pytest在导入TorchDynamo时尝试向只读依赖目录写pyc，造成等待。用`python -B -m pytest`解决，未跳过模型训练测试。
- 本地旧Anaconda的numexpr和bottleneck版本产生两项pandas可选加速器警告。测试通过；服务器建议使用README中的新环境，不复用本机旧环境。
- 未运行独立静态类型检查器或在线依赖漏洞扫描，不能将源码编译通过解释为这些检查已经完成。

## 必须留到服务器验证

1. 真实LLM端点的参数兼容性、输出质量、tokenizer与usage一致性。
2. BAAI/bge-m3完整权重加载、语义编码效果与显存。
3. Linux/CUDA运行、训练峰值显存、完整表的吞吐和总耗时。
4. 六表、五随机种子的完整精度/召回/F1及与修复后旧框架的配对比较。
5. 关系门控、来源去重、反事实、文本表示及伪标签的消融效果。

本地验证说明已覆盖场景中的接口、数据流和训练过程可用，不承诺无未知缺陷，也不证明研究目标已经达到。尤其是错误分数未用独立真实标签校准，真实LLM伪标签仍可能含噪。
