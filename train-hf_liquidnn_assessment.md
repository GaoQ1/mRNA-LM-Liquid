# train-hf（halflife）迁移液态神经网络评估与计划

## 1. 当前 `train-hf` 训练链路（基于现有代码）

### 1.1 调用入口
- `Makefile` 的 `train-hf` 目标执行：`python finetune_all.py --task halflife --output models/halflife --device 0`（`Makefile:16-17`）。

### 1.2 训练脚本行为
- 任务分支：`halflife -> build_saluki_dataset(0)`（`finetune_all.py:76-77`）。
- 模型：`FullModel`，由 3 个预训练 BERT 编码器（5'UTR/3'UTR/CDS）+ LoRA + 回归头组成（`FullModel.py:32-35`, `FullModel.py:50-70`, `FullModel.py:171-180`）。
- tokenization：UTR 使用 1-mer，CDS 使用 3-mer（`dataload.py:127-129` 与 `FullModel.py:239-280`）。
- 指标：回归任务计算 Pearson/Spearman（`finetune_all.py:111-125`）。

### 1.3 数据规模（`data/mrna_half-life.csv`）
- 总样本：12919
- 训练集 split 0-7：10333
- 验证集 split 8：1293
- 测试集 split 9：1293

### 1.4 当前基线的工程问题（先修再谈架构迁移）
- `--batch` 参数定义了，但训练实际写死 `per_device_train_batch_size=12`（`finetune_all.py:31`, `finetune_all.py:98`）。
- `metric_for_best_model` 变量计算了，但未传入 `TrainingArguments`，`load_best_model_at_end` 实际不按 Spearman 选最佳（`finetune_all.py:49-56`, `finetune_all.py:90-109`）。
- `--output models/halflife` 仅用于 HF checkpoint，最终权重固定写到 `outputs/halflife/finetuned_model.pt`（`finetune_all.py:93`, `finetune_all.py:150`），与参数语义不一致。

## 2. 是否有必要改成液态神经网络（LNN/LTC/CfC）

## 结论
**短期不建议把 `train-hf` 全量改成液态神经网络主干。**

## 原因
- 当前任务是“静态序列 -> 单值回归”，每条样本没有显式时间轨迹；液态网络的核心优势在连续时间/不规则采样时序建模，这里匹配度不高。
- 现有方案强依赖 3 个 RNA 预训练编码器。若全替换为液态网络，等于放弃当前预训练迁移收益，在 1 万级训练样本上风险较高。
- 项目当前主要瓶颈先是训练/评估流程一致性（上面的 3 个工程问题），不是明显的“模型动态性不足”。

## 3. 优劣对比（当前方案 vs 液态网络）

| 维度 | 当前方案（3xBERT + LoRA + MLP） | 液态网络（LTC/CfC） |
|---|---|---|
| 任务匹配度 | 对序列语义建模强，且与现有预训练完全匹配 | 更擅长连续时间动态系统；本任务收益不确定 |
| 数据效率 | 预训练迁移 + LoRA，数据效率高 | 若从头训练，通常更依赖任务数据与调参 |
| 长程依赖 | Transformer 对长程模式表达强 | 递归/连续时间建模可表达动态，但不一定优于现有预训练特征 |
| 工程改造成本 | 低（维持现状） | 高（新增网络、训练循环、超参体系、稳定性验证） |
| 训练稳定性 | 当前框架成熟（HF Trainer + PEFT） | ODE/LTC 训练对步长、初始化更敏感；调参成本高 |
| 推理/部署 | 与现有 API/模型路径兼容 | 需新增依赖与部署验证，兼容成本上升 |
| 可解释性 | 中等（可做 attention/embedding 分析） | 某些 liquid 结构可解释性更好，但需要额外分析工具 |

## 4. 建议执行计划（低风险、可回滚）

### 阶段 0：先修基线一致性（必须）
1. 让 `--batch` 真正生效。
2. 把 `metric_for_best_model='spearmanr'` 和 `greater_is_better=True` 接入 `TrainingArguments`。
3. 统一最终模型保存路径（跟 `--output` 对齐），避免 `models/` 与 `outputs/` 语义混用。
4. 固定随机种子，跑 3 次基线，记录 `val/test spearman` 均值与方差。

### 阶段 1：液态网络最小可行 PoC（建议“只替换头部”，不动预训练主干）
1. 冻结 3 个预训练编码器，只取其 pooled embedding。
2. 把当前 MLP 回归头替换成小型 CfC/LTC 头（输入维度 2304，hidden 64/128）。
3. 与当前 head 在同数据切分、同训练轮数下做 A/B。

### 阶段 2：Go/No-Go 判定
满足以下条件才继续深挖液态方案：
1. `val spearman` 相对基线提升 >= 0.02。
2. `test spearman` 不退化（允许波动 <= 0.01）。
3. 训练时间不超过基线 1.3 倍。
4. 3 次不同随机种子下结果方向一致。

### 阶段 3：若 PoC 通过，再考虑深度改造
1. 尝试将 token-level 序列输入 liquid 层（而非只替换 head）。
2. 增加不确定性评估与 OOD 测试，确认泛化收益。
3. 评估 API 推理延迟和显存占用，决定是否进入生产。

## 5. 最终建议
- **现在不建议直接把 `train-hf` 改成“全液态神经网络”。**
- **建议先把现有训练基线修到一致可复现，再做“液态头部 PoC”验证。**
- 若 PoC 达不到上面的 Go 标准，应继续沿当前预训练+LoRA 路线优化（数据清洗、损失函数、采样策略、超参搜索），投入产出比更高。
