# Finding the Frame Workshop — 论文适合性深度分析

## 一、Workshop 要求对比

| 要求 | 你的论文 | 状态 |
|------|---------|------|
| **主题匹配** — "rethinking core aspects of RL paradigm" | 质疑"规范时间编码"假设 | ✅ 完美匹配 |
| **页数限制** — 审稿版7页(不含参考文献/附录)，终版8页 | 当前编译8页 | ⚠️ 审稿版需砍1页 |
| **模板** — RLC author kit (rlj.sty) | 使用rlj.sty | ✅ |
| **匿名** — double-blind | 已匿名 | ✅ |
| **截止日期** — 延至2026-05-29 AoE | 今天5-28 | 🔴 明天截止！ |
| **未发表** — 不接受已发表工作 | 全新工作 | ✅ |

## 二、与2025年录用论文对比

### 2025年录用论文（talks）

1. **"Thinking is Another Form of Control"** (最佳论文) — 重新定义thinking在RL中的角色
2. **"Analogy making as amortised model construction"** — 类比推理作为模型构建
3. **"Agent-centric learning"** — 从外部reward到内部知识管理

### 你的论文定位

你的论文直接挑战了一个RL中的隐含假设：**recurrent agent会学到唯一的/规范的时间编码**。这是一个"examining the conceptual foundations of RL"的典型问题，与workshop主题高度一致。

**对比分析：**

| 维度 | 2025录用论文均值 | 你的论文 |
|------|----------------|---------|
| 概念新颖性 | 高（重新定义RL概念） | 中高（质疑时间表征唯一性） |
| 实验深度 | 中等（多为理论/概念性） | 高（5个seed、4个horizon、3个环境） |
| 写作质量 | 高 | 中高（诚实报告，但叙事略弱） |
| 引用量 | 15-30篇 | 14篇（偏少） |
| 页数 | 7页 | 8页（超1页） |

## 三、论文优势

1. **主题完美匹配** — "Evidence Against a Canonical Temporal Code"直接质疑RL中的隐含假设
2. **诚实报告** — 承认失败seed、single-neuron overlap at chance、CKA>0.96的矛盾
3. **数据透明** — 全部数据可追溯到JSON文件，代码已开源
4. **多维度分析** — CKA、linear probe、PCA、MI、single-neuron、cross-seed CKA

## 四、论文弱点（需改进）

### 关键问题

| # | 问题 | 严重性 | 修复建议 |
|---|------|--------|---------|
| 1 | **超页数** — 审稿版限7页，当前8页 | 🔴 | 砍掉Appendix A（single-neuron at chance）+ 精简Discussion |
| 2 | **标题夸大** — "Evidence Against"过强 | 🟡 | 改为"Rethinking the Canonical Temporal Code"或类似 |
| 3 | **引用偏少** — 14篇，缺失关键文献 | 🟡 | 补充Dasgupta et al., Schmucker et al.等3-5篇 |
| 4 | **5个seed不够** — 对强claim统计力不足 | 🟡 | 弱化claim，加更多hedge |
| 5 | **单环境** — 只有DoorGrid的深度分析 | 🟡 | 在Discussion中诚实地承认 |

### 需要删除/精简的内容（为砍到7页）

- **Appendix A** (single-neuron analysis) — 已在正文承认at chance，可以删除
- **MLP baseline** — 可以缩减为1句话
- **Table 1** — 可以合并到正文叙述中
- **Q3 horizon analysis** — 可以精简T=16/32的描述

## 五、综合评估

### 录用概率评估

| 因素 | 权重 | 得分 | 说明 |
|------|------|------|------|
| 主题匹配度 | 25% | 9/10 | 完美匹配workshop主题 |
| 概念新颖性 | 25% | 7/10 | 质疑隐含假设，但不是全新的paradigm |
| 实验质量 | 20% | 7/10 | 方法合理但5个seed偏少 |
| 写作质量 | 15% | 7/10 | 诚实但叙事可以更强 |
| 引用完整性 | 15% | 5/10 | 14篇偏少，缺失关键文献 |
| **加权总分** | | **7.2/10** | |

### 结论

**论文处于"borderline accept"水平。** 主题匹配度是最大的优势——这正是Finding the Frame workshop想要的工作。但有几个关键问题需要在提交前解决：

1. **必须砍到7页** — 这是硬性要求
2. **标题需要弱化** — "Rethinking"比"Evidence Against"更安全
3. **补充3-5篇关键引用** — 特别是Dasgupta et al.和Schmucker et al.
4. **讨论部分需要加强** — 把"plurality"的nuance讲清楚

### 截止日期警告

⚠️ **截止日期是明天（2026-05-29 AoE）！** 如果要提交，今晚必须完成所有修改。

## 六、优先修改清单（按ROI排序）

1. **砍到7页** — 删除Appendix A + 精简MLP/horizon描述
2. **弱化标题** — "Do Recurrent RL Agents Discover a Unique Internal Clock? Rethinking Temporal Codes in RL"
3. **补充引用** — Dasgupta et al. (2020), Schmucker et al. (2023), Saxe et al. (2013)
4. **精简Discussion** — 聚焦"degeneracy vs redundancy"的nuance
5. **重新编译** — 确认7页
