# 输入支持边界与逐模型标准化审计结论

本次在原始 C/D 事件和已保存评分数组上只读核查；未修改冻结源码、协议、模型或正式数据，未训练、推理或重新评分。

| C/D | 发射事件 | eligible | 原始到 float32 支持碰撞 | 最小 p32−y32 (GeV/c) | 最小向下 ULP 比 |
|---|---:|---:|---:|---:|---:|
| C | 500000 | 499285 | 0 | 0.2957799434661865 | 8284017.5 |
| D | 500000 | 499275 | 0 | 0.2978813648223877 | 8311172 |

每个 block 的六个正式评分缓存数组与从原始 ledger 独立重建的值、字节及 UID 顺序完全一致；42 组计数也一致。eligibility 使用已保存 entry/exit 几何与退出动量向量重新判断，没有仅按评分成功行筛选。y64=p64−pout64 与原目标逐位一致，退出动量与 exit vector 范数逐位一致。

zero、negative、nonfinite 和所有预固定 small-positive 阈值计数均为零。这里 p−y 是退出动量 p_out，实际正式样本远离 y=p 的输入支持边界。该结论覆盖已观察到的正式 C/D 样本，不是所有未来输入的精度保证。

进一步检查全部 240 registered runs × C/D：每 block 14 种已保存 preprocessor，独立计算纯中心与标准化，再对每个模型保存的 z/b 数组逐元素比较。共 960 份数组、479,308,800 个 float32 数值，逐位差异为零、最大 ULP 差为零；全部 z<b，最小 b−z=751.7498073577881。去重只节省同一预处理的计算，没有少检查模型文件。

此重算使用冻结 RepresentationPreprocessor 的纯数学方法与真实 torch.float32 算式，建立保存 z/b 与事件顺序的直接联系；它不是中心物理公式的独立理论验证，也没有重算 neural flow/log-score。240 是 registered run IDs，并非 240 个独立或数值不同的预测器。

保存的其余 likelihood/support/PIT 组件由 support_terms 独立全模型审计；本报告不能替代该项。原始 first-visit 信息取自 ledger，未重新遍历 Geant4 step 历史。

与另一份仅缓存输入审计的当前版本相比，数量、所有支持符号计数、最小值/最大值、最小 ULP 比及全部共同分位数均完全一致。该独立核验最初草稿采用 float32 插值；保留草稿后，当前版本已明确先升 float64，与本审计预设定义一致。

完整数值：input_margin_review.json、input_margin_by_momentum_bin.csv、standardization_replay.json、standardization_replay.csv。这些输出只含汇总、model IDs 与 provenance 哈希，没有事件 UID 或事件行。
