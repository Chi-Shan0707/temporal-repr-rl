# 综合审查报告 — temporal_repr 项目

**审查日期:** 2026-05-28
**审查范围:** 全部代码 (46个Python文件, 6250行) + 论文 (360行tex) + 数据 (24个JSON)

---

## 一、数据完整性审计

### 已验证 ✅

| 论文声明 | 实际数据 | 来源文件 | 状态 |
|---------|---------|---------|------|
| Table 2: raw=0.40±0.10 | 0.40±0.10 | 5个seed的results.json | ✅ |
| Table 2: sin_PE=0.63±0.09 | 0.63±0.09 | 同上 | ✅ |
| Table 2: phase=0.28±0.21 | 0.28±0.21 | 同上 | ✅ |
| Table 2: onehot=0.27±0.16 | 0.27±0.16 | 同上 | ✅ |
| Table 3: T=8 sin=0.58±0.08, raw=0.56±0.14 | 匹配 | 11_lstm_dg_t8_seed* | ✅ |
| Table 3: T=16 sin=0.60±0.10, raw=0.64±0.15 | 匹配 | scaling_dg_t16_seed* | ✅ |
| Table 3: T=32 sin=0.50±0.08, raw=0.70±0.22 | 匹配 | scaling_dg_t32_seed* | ✅ |
| 5 seeds Return=0.920 | best_eval均为0.920 | train_log.json | ✅ |
| Cross-seed CKA >0.96 | off-diag 0.962~0.999 | cross_seed_cka_matrix.json | ✅ |
| Untrained: R²=0.30, phase_acc=0.30 | R²=0.304, acc=0.298 | untrained_lstm_doorgrid | ✅ |
| Untrained: CKA≈0.15 | best=0.153 (raw_scalar) | 同上 | ✅ |
| 4/5 seeds best CKA with sin_PE | 42:0.74, 123:0.64, 789:0.66, 1024:0.65 | 各seed results.json | ✅ |
| Seed 456 best with phase | phase=0.62, sin_PE=0.46 | seed456 results.json | ✅ |

### 发现问题 ⚠️

| 问题 | 严重性 | 详情 |
|------|--------|------|
| **MLP DoorGrid数据缺失** | 🔴 严重 | 论文L168声称"MLP baseline fails to learn (Return=-2.0, R²<0.04)"，但磁盘上**不存在MLP DoorGrid实验结果**。只有MLP FrozenLake (08/09)。`run_training.py`和`run_multiseed.py`都没有定义MLP DoorGrid实验。 |
| **T=16有1个seed失败未报告** | 🟡 中等 | seed 1024在T=16时best_eval=-2.0（完全失败），但论文只说"2/5 seeds fail at T=32"，未提T=16也有失败。Table 3的T=16 Return=0.800†用了†标记但脚注未明确说明1个seed失败。 |
| **T=32 seed 123不稳定** | 🟢 轻微 | seed 123在T=32时best_eval=0.64但final_avg=-2.0，说明训练后期崩溃。CKA分析用的是best.pt，所以数据有效，但应注明。 |

---

## 二、代码审查

### 严重问题 🔴

**1. `train.py:117` — MLP训练路径会崩溃**

```python
# Line 114-119 (MLP path):
logits, values, _, _ = model(obs_tensor)
dist = torch.distributions.Categorical(logits=logits)
log_probs_list = [dist.log_prob(act_tensor[t].squeeze())[0] for t in range(steps)]
```

当batch_size=1时，`act_tensor[t].squeeze()`是0维tensor，对其做`[0]`索引会抛出`IndexError`。LSTM路径(lines 107-113)没有这个问题因为每次只处理单步。

**修复:** 移除`[0]`，或者在squeeze前确保维度正确。

**2. `train.py:20-21` — 硬编码绝对路径**

```python
sys.path.insert(0, "/mnt/d/CS/ReinforcementLearning/undo_gap/temporal_repr")
```

所有文件都有此问题（train.py, env_factory.py, wrappers.py, run_high_roi.py等），导致项目**不可移植**。

### 中等问题 🟡

**3. `linear_probe.py` — 无交叉验证**

单次train_test_split (80/20)给出的R²和accuracy是点估计。应使用5-fold CV报告mean±std。

**4. `fourier.py:81` — 功率归一化错误**

`np.abs(rfft(ts)) ** 2 / n` — 应除以`sum(window**2)`而非`n`。不影响频率识别但绝对功率值偏差~2.67倍。

**5. `mutual_info.py` — 分箱MI稀疏性**

3个PCA维度×10个bin = 1000个joint bin，样本量不足时大部分bin为空，导致MI高估。KSG估计器是正确的。

**6. `compute_cross_seed_cka.py` — mean-hidden CKA仅4个点**

对(T=4, 64)矩阵做CKA，n=4恰好是unbiased HSIC的最低要求，统计上极不可靠。

### 轻微问题 🟢

**7. `pca_vis.py` — 对phase用Pearson相关不恰当**

phase是循环变量(0→P-1→0)，Pearson r会低估真实关联。应用circular correlation。

**8. 实验管理文档过时**

- `PROGRESS_REPORT.md`标记多个实验"未启动"但实际已完成
- `EXPERIMENT_CONCLUSIONS.md`的"下一步行动"与`REVISION_STATUS.md`矛盾
- `REVISION_STATUS.md`标题是"Temporal Encoding...Degrades"但当前论文标题是"Do Recurrent RL Agents..."

**9. 超参数未随结果保存**

learning rate, entropy_coef, hidden_dim, grad_clip等硬编码在train.py中，未写入结果JSON。

---

## 三、论文审查

### 结构与叙事

- **Q1/Q2/Q3框架清晰**，逻辑流畅
- **标题"Evidence Against a Canonical Temporal Code"** 略微夸大：cross-seed CKA>0.96说明表征高度相似，"plurality"体现在与候选编码的对齐差异上，而非表征本身不同
- **14篇参考文献偏少**，缺失Dasgupta et al.、Schmucker et al.等关键文献

### 关键引用缺失

- Dasgupta et al. (2020) — RL中的表征学习
- Schmucker et al. (2023) — 循环RL agent的时间表征
- Saxe et al. (2013) — 随机初始化对表征的影响

### 统计严谨性

- 5个seed对强claim不够充分（CV of std本身高度不确定）
- 无permutation test或bootstrap CI来验证seed间CKA差异的显著性
- single-neuron overlap分析(1.8/8≈chance 1.88/8)诚实但浪费篇幅

---

## 四、优先修复清单

| 优先级 | 问题 | 行动 |
|--------|------|------|
| P0 | MLP DoorGrid数据缺失 | 运行实验获取真实数据 |
| P0 | train.py MLP路径bug | 修复line 117的索引错误 |
| P1 | T=16失败seed未报告 | 论文中补充说明 |
| P1 | 硬编码绝对路径 | 改用Path(__file__).parent.parent |
| P2 | 线性探测无CV | 添加5-fold CV |
| P2 | 超参数未记录 | 在train()中dump config到JSON |
| P3 | 文档过时 | 删除或更新PROGRESS_REPORT等 |
