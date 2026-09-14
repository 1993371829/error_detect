# Hypergraph ED

独立的表格错误检测研究实现：LLM诱导受限语义约束超图，共享注意力模型执行条件预测，来源感知融合与独立上下文反事实检查负责判断和定位。全程不要求人工标注。

## 快速运行

推荐服务器 Python 3.11、Linux和CUDA。支持Python 3.9–3.12；CPU适合小规模测试。首先按服务器CUDA版本安装PyTorch，再安装项目：

```bash
python -m venv .venv
source .venv/bin/activate
# 按 https://pytorch.org/get-started/locally/ 选择匹配驱动的PyTorch安装命令。
pip install -e '.[semantic,test]'
python -B -m pytest -q
```

所有源码均在src/hypergraph_ed，不导入原项目stage_1、stage_2、stage_3或paths。上传整个refactored_framework文件夹即可；数据、密钥和模型权重分别提供。无需上传outputs和本地缓存。

本地完全离线、真实CPU训练的样例：

```bash
pip install -e '.[test]'
python -m hypergraph_ed doctor --config configs/local_smoke.yaml
python -m hypergraph_ed run --input tests/fixtures/tiny_dirty.csv --config configs/local_smoke.yaml --output outputs/smoke
python -m hypergraph_ed evaluate --dirty tests/fixtures/tiny_dirty.csv --clean tests/fixtures/tiny_clean.csv --predictions outputs/smoke/detections.csv
# 相同输入与配置可恢复；已完成的运行直接返回原报告。
python -m hypergraph_ed run --input tests/fixtures/tiny_dirty.csv --config configs/local_smoke.yaml --output outputs/smoke --resume
```

local_smoke使用**固定LLM JSON响应和真实PyTorch模型**。词法编码是明确的消融配置，并不模拟BGE效果。两轮训练只验证流程，输出F1不能用于研究结论。Windows可执行 `powershell -File scripts/run_local_checks.ps1`。

## 服务器模型与API配置

准备冻结的语义模型（该命令需要访问Hugging Face，但不发LLM请求）：

```bash
python -m hypergraph_ed prepare-model --model-id BAAI/bge-m3 --destination /models/bge-m3
export SEMANTIC_MODEL_PATH=/models/bge-m3
export LLM_TOKENIZER_PATH=/models/your-deployed-model-tokenizer
export LLM_API_KEY='your-key'
export LLM_MODEL='your-deployed-model-name'
export LLM_BASE_URL='https://your-provider/compatible-mode/v1'
python -m hypergraph_ed doctor --config configs/server.yaml
bash scripts/run_server.sh /data/movies_dirty.csv /results/movies
```

`.env.example`只列出变量，不会自动读取父目录.env。LLM tokenizer必须与部署服务使用的聊天模板及分词方式相匹配；准备方式取决于服务商对应模型。不能可靠计数时客户端不发请求，继续本地流程。BGE或CUDA缺失时server配置明确失败，不静默变换算法。`doctor`默认只做本地检查，不进行付费探测。

OpenAI兼容客户端关闭SDK自动重试。Qwen/DashScope和DeepSeek官方端点按接口传入关闭思考的参数；其他服务使用其原生max_tokens限制。生产前需在服务器核对服务商usage与本地计数：一旦返回用量超过预留，账本记录差异并禁止后续请求。框架不能替服务商保证其返回用量和tokenizer契约正确。

## 预算与恢复

每张表、每次独立冷启动最多20次请求、100000个输入+输出token。结构6次/40000，审核4次/20000，伪标签8次/30000，重试2次/10000。不需要用完。伪标签最多128次被请求的单元格判断，重复请求也计数。

预算先写预留再发请求，崩溃留下的pending请求保留完整费用。未知usage失败按预留计费。已缓存的成功响应不新计请求。OS文件锁禁止并行进程操作同一账本；恢复不得改变输入、配置或语义模型资源。GPU模型检查点保存模型、优化器、早停状态和关系质量。检查点是本项目生成的可信产物，不接收任意外部检查点。

全流程只允许一轮结构修订和教师标签增量训练。预算耗尽后本地完成，并在report.json记录原因；不产生人工待办。

## 数据结构与实现

- `ConstraintProgram`保存字段类型/单位、约束、参与字段、目标字段、条件、注册算子和来源。最多96条关系，每个目标最多6条，每条最多4个参与字段、2个条件字段。
- 支持missing、dmv、type、regex、duplicate、normalize、range、statistical、neighbor、fd、compare、temporal、arithmetic。CFD由FD加conditions表示。字符串标准化为映射/大小写/空白处理。正则限制为短且无分支/分组的表达式；更复杂格式需要新增注册算子。
- 算术是AST白名单的残差表达式；不允许调用函数、属性访问或任意Python代码。单位不做隐式转换；归一化需显式定义。Duration不再匹配ratio。
- 本地结构挖掘提供保守起点；LLM分别查看典型、稀有和冲突视角，返回可编译结构。相同逻辑先规范化去重。
- 共享模型用目标字段查询相关上下文。原值不进入本单元格的条件预测输入；其观测表示只在错误判别头与预测结果比较。类别身份使用精确词表嵌入；高基数文本使用字符特征以及server模式下的冻结BGE表示。
- 三折按完整行指纹划分，重复行同折。编码统计、词表和同组参考来自训练折；每格使用未训练该行的模型打分。LLM结构使用全表无标签画像，因此属于表内转导设置，不宣称完全归纳式评估。
- 目标有效性与上下文可靠性分开。明确格式损坏的目标不参加重构；上下文使用软可靠性衰减。未检出错误不等于正常金标签。
- 关系门控特征为覆盖、支持、bootstrap稳定性、独立上下文反证、节点消融预测增益。增益在外层训练折的验证样例上估计，不使用clean表。
- 损失为masked + 0.5 synthetic + 0.1 teacher。原始样本弱正常权重0.25；合成扰动不是现实错误的完备模型。标签冲突直接丢弃。训练早停依据无真值的重构验证损失。
- 同源证据规范化去重后，按7个证据族归一化池化，经共享门控和错误头组合。局部check_passed不等同于整体clean投票。
- 反事实定位使用关系外字段的同组记录、其他来源检查和本地预测候选。单字段改动能解释冲突才定位；多字段差异和相互冲突的上下文保持未决。该机制提供预测性定位证据，不是因果证明，也不会修改原表。

## 输出与评估

`detections.csv`覆盖所有单元格，包含is_error、error_score、error_type、low_confidence、model_residual、scoring_fold、relations、evidence、localization、suggested_fix。默认判错阈值0.5，0.35–0.65或证据不足标为低置信；分数未经真实标签校准。

输出目录还有program_initial.json、program.json、profiles.json、manifest.json、budget.json、report.json、models_round0/及可选models_round1/、refinement.json和LLM响应缓存。原始表不可变，修复建议不自动应用。

`evaluate`是独立命令，检查预测完整性，并按现有基准的缺失哨兵规范比较dirty/clean，报告总体和逐列P/R/F1、低置信比例。训练接口不接受clean路径。错误类型的真实召回需要额外提供类型标注，本项目不会用预测类型冒充真值类型。

六表/五随机种子完整运行：

```bash
python scripts/run_batch.py --data-dir /data --output /results/full --config configs/server.yaml --evaluate
```

**这会产生30次独立运行，理论上限600次请求、300万token**，不是整个批次共享10万token。每次运行独立账本，支持恢复。summary.json记录每次结果；aggregate.json给出按数据集bootstrap的描述性区间，不冒充相对旧框架的配对显著性。

本地真实表格式冒烟：

```bash
python scripts/run_batch.py --data-dir /path/to/data --output outputs/six_table_smoke --config configs/local_smoke.yaml --seeds 42 --sample-rows 32
```

## 验证边界与研究目标

详见VALIDATION.md。本地验证CPU训练、预算状态机、模型恢复、结构隔离、真值隔离、异常输入、CLI和独立安装。真实LLM、BGE完整权重、CUDA显存和完整数据集效果需在服务器验证。

相对修复后的旧框架提高宏平均F1至少2个百分点是研究目标，不是代码交付承诺。伪标签噪声、共同错误、合成扰动偏差和关系覆盖不足都可能降低效果；必须用消融和统一模型/预算实验判断设计价值。不要把两轮smoke训练结果或fixture的模拟token数写成真实API性能结果。
