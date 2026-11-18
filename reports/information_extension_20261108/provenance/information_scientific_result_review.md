# 新 C/D 科学结果独立解读与正文复核

审阅时间：2026-11-07T20:40:11.816960+00:00. 本笔记只读取已完成结果、独立验算记录和正文；未读取原始事件、额外推理、训练或修改冻结工件。下文 Δ=mean_seed,event(log p_C−log p_Raw)，正值表示物理中心平均更好。

**核心判断：新证据支持有限设计内非单调的 representation × conditioner-information 交互，而不是“越缺信息越有利于物理先验”的单调规则。四个信息层级平均均有收益；Coarse2 相对 Raw 的收益最大，但 Full5 绝对平均 NLL 最好。局部损失在新数据的预设分组和多重比较带下仍存在。它们不是信息梯度上的符号翻转阈值，也不能恢复旧错误中心的全局 Hidden 失败结论。**

C/D 各发射500,000事件，S1分别499,285和499,275。全部主区间均条件于这五对冻结模型及既定生成设置；不包含重新抽训练集、重新训练、优化器种子总体或物理模型的不确定性。C/D 分开分析，下面95%同时带不是两个catalog合起来的联合95%保证。

**主16对照。** 所有8 gain在每catalog的16对照同时带均高于0；三个信息递增的邻接差依次为正、负、负，两条endpoint Full5−Hidden5均负。表中数值为estimate [16族同时带]，单位nat/event。

| 对照 | C | D |
|---|---:|---:|
| gain:Hidden5/f010 | +0.006272 [+0.005709, +0.006835] | +0.006401 [+0.005830, +0.006971] |
| gain:Coarse2/f010 | +0.010399 [+0.009844, +0.010954] | +0.010623 [+0.010059, +0.011186] |
| gain:Coarse8/f010 | +0.003082 [+0.002741, +0.003423] | +0.003320 [+0.002979, +0.003662] |
| gain:Full5/f010 | +0.001845 [+0.001580, +0.002110] | +0.001876 [+0.001609, +0.002142] |
| gain:Hidden5/f100 | +0.006233 [+0.005661, +0.006804] | +0.006543 [+0.005962, +0.007124] |
| gain:Coarse2/f100 | +0.010895 [+0.010287, +0.011502] | +0.011196 [+0.010576, +0.011816] |
| gain:Coarse8/f100 | +0.002987 [+0.002653, +0.003321] | +0.003332 [+0.002989, +0.003676] |
| gain:Full5/f100 | +0.001550 [+0.001300, +0.001799] | +0.001659 [+0.001408, +0.001911] |
| gain_change:Coarse2_minus_Hidden5/f010 | +0.004127 [+0.003838, +0.004417] | +0.004222 [+0.003922, +0.004522] |
| gain_change:Coarse8_minus_Coarse2/f010 | -0.007317 [-0.007780, -0.006854] | -0.007302 [-0.007776, -0.006828] |
| gain_change:Full5_minus_Coarse8/f010 | -0.001237 [-0.001496, -0.000978] | -0.001445 [-0.001707, -0.001182] |
| gain_change:Full5_minus_Hidden5/f010 | -0.004427 [-0.004975, -0.003879] | -0.004525 [-0.005084, -0.003966] |
| gain_change:Coarse2_minus_Hidden5/f100 | +0.004662 [+0.004395, +0.004929] | +0.004653 [+0.004386, +0.004920] |
| gain_change:Coarse8_minus_Coarse2/f100 | -0.007908 [-0.008447, -0.007368] | -0.007864 [-0.008417, -0.007310] |
| gain_change:Full5_minus_Coarse8/f100 | -0.001437 [-0.001656, -0.001219] | -0.001673 [-0.001901, -0.001445] |
| gain_change:Full5_minus_Hidden5/f100 | -0.004683 [-0.005204, -0.004162] | -0.004884 [-0.005415, -0.004352] |

种子例外不能被窄事件区间掩盖：Full5/f010的seed3102在C/D分别为−0.000619034459、−0.000491149500；Full5−Coarse8/f010的seed3104却分别为+0.001645361588、+0.001255408336，尽管该对照五seed平均为负。其余七种gain设置全部五seed为正；Coarse2−Hidden5、Coarse8−Coarse2、Full5−Hidden5在两个比例和两catalog下所有seed方向均与均值一致。
来源：[C主区间](<E:/JINST REP/_research_extension_20261108/fresh_analysis/C/primary_uncertainty.csv>)、[D主区间](<E:/JINST REP/_research_extension_20261108/fresh_analysis/D/primary_uncertainty.csv>)，配套primary_seed_contrasts.csv与primary_contrast_summary.csv。

**绝对表现与实践量级。** Raw和CorrectedPhysics的平均绝对NLL都按Hidden5→Coarse2→Coarse8→Full5严格降低，四个catalog×fraction条件一致；这是描述性绝对比较，不能把主16族的gain区间移用于它。Corrected Hidden5比Full5差0.0190017208–0.0195781353；Coarse2差0.0050038671–0.0053290510；Coarse8差0.0003316612–0.0004879729。Corrected的Full5对Coarse8绝对优势并非所有seed成立：C10/C100/D10/D100分别仅4/3/2/4个seed偏向Full5。
固定20k更新下10%模型的平均绝对NLL在这些条件均优于100%，不能解释为少数据更优或统计效率定律；每事件训练曝光与checkpoint选择同时参与该有限预算比较。旧Full矩阵在C/D的24个Small−Large平均绝对NLL也全部<0，但容量的representation-gain交互仍混合。
Full5平均gain对应约0.1551%–0.1877%的几何平均密度比增加；Coarse2约1.0453%–1.1259%。不是精度百分比、事件改善比例、均匀提升或下游效用。五seed平均下NΔ是五对模型各自累计logscore差的平均，不是mixture-density得分。来源：各block model_summary.csv和primary_uncertainty.csv。

**局部损失的fresh证据。** 以下Mean<0为描述性负组数，Upper<0采用该catalog全部8×42=336组的同时带。

| 条件 | C Mean<0 / Upper<0 | D Mean<0 / Upper<0 |
|---|---:|---:|
| Hidden5/f010 | 25 / 23 | 27 / 22 |
| Hidden5/f100 | 22 / 20 | 23 / 20 |
| Coarse2/f010 | 18 / 9 | 17 / 9 |
| Coarse2/f100 | 13 / 10 | 14 / 10 |
| Coarse8/f010 | 2 / 0 | 3 / 0 |
| Coarse8/f100 | 6 / 1 | 5 / 1 |
| Full5/f010 | 3 / 0 | 1 / 0 |
| Full5/f100 | 9 / 0 | 9 / 1 |

Hidden5的负均值组含eligible事件比例分别为C10=59.5363%、C100=52.3881%、D10=64.2962%、D100=54.7636%；这不是这么多单个事件变差。最低p=[0.3,0.4)的Hidden5均值收益为0.090528–0.094793，抵消[0.4,0.6)、[0.8,1.0]等区间负均值：总体收益与多数负组可同时成立。两catalog在10%共有22个相同组、100%共有20个相同组各自336带上界<0；这只是分别通过各自族区间的重复模式，不新增一个跨catalog联合置信声明。
可以称为“对早期探索性局部trade-off的预先规定fresh follow-up，支持固定模型在这些组上的局部平均损失”；不能称为旧24/42计数被原样确认：旧模型Hidden为2输入，新模型Hidden5有匹配5输入架构，当前负均值计数22–27。旧A/B仍是post-hoc且有历史用途。也不能仅用这些负组计数证明连续物理区域内处处失败、个体事件受害率或information-compatibility阈值。来源：group_uncertainty.csv、group_seed_variation.csv，全部336组和五seed值均保留。

**机制支持与限制。** Source56共同细分的事件加权位置方差（×10^-10 (GeV/c)^2）为：

| 可见信息 | Raw | Corrected | Raw−Corrected |
|---|---:|---:|
| Hidden5 | 1.576797553 | 1.221880110 | 0.354917443 |
| Coarse2 | 1.542582532 | 1.175761877 | 0.366820655 |
| Coarse8 | 0.930537675 | 0.905750845 | 0.024786830 |

每种representation自身burden随信息细化下降，而它们的差在Coarse2略高、Coarse8明显更小，与gain的宽泛形状相符。该量是同一reserved source pool的分箱描述，不是对真实条件均值变化的无误差估计，也没有对应误差条；两fraction原单位结果重复是global μ消去，不是新重复实验。Full精确p的hidden-mean burden理论上为0，但Full仍有有限flow gain，这直接限制了把位置诊断当完整NLL机制。
四项NLL账目正确、原单位score重建受float32舍入限制。新80各组的paired log_s与support对比均为0；个别模型density重建最大误差2.3841858e-6 nat，paired分解最大1.9073486e-6。增益由base-coordinate与spline-Jacobian项相抵后得到，不能分别命名为location与scale/shape因果贡献。例：C/Coarse2/f100为−0.01671335+0.02760785=+0.01089450；同context的f010却是+0.00452570+0.00587331=+0.01039901。分量方向随训练比例变化进一步说明它们不是稳定独立的物理归因。
PIT提供具体、但非正交的局部校准描述。100% Hidden5在p=[0.3,0.4)的预测median coverage由Raw的C/D 0.319452/0.317230改善到0.398953/0.397828（理想0.5，但仍偏离）；在p=[0.9,1.0]则由0.558553/0.553472变为0.587485/0.581520，离0.5更远，同时predictive-IQR coverage由约0.478–0.479降到约0.435（理想0.5），lower5%频率由约0.067–0.069升到0.102–0.104。Coarse2在最低p的median/IQR/tail指标也明显更接近参考值，但高p可更差。这与局部score trade-off相容，不能证明全部分布校准、唯一location机制，或把median/IQR/tail的改善分配成NLL贡献。以上为已预设PIT量的例示性结果解读，非新增检验族。
来源：source_analysis_retry1/source_location_diagnostics.json；fresh各block density_component_diagnostics.json、density_component_seed_summary.csv、predictive_diagnostics.json、residual_diagnostics.json。

**旧160对照。** C/D各6个Full容量×fraction下：Raw−Corrected均正、Learned−Corrected均正、Simple−Raw均负，因此Simple也有平均收益，不能将总体改善全部归为物理先验独特性。对应均值范围：Raw−Corrected 0.0013976–0.0023482；Learned−Corrected 0.0011054–0.0032343；Simple−Raw −0.0011994至−0.0006299。Large/f100的Learned−Corrected仍有seed3105负例（C−0.00072972、D−0.00087386）；FullSmall/f010的Raw−Corrected仍有seed3102负例。Simple在多个设置也有seed反向。Simple/Learned没有新ladder对应拟合，不能由此断言Coarse2时physics优于它们。
容量/比例gain交互仍非普遍规律：Small−Large/f010在C为+0.00005453、D为−0.00002656，f030/f100平均为负而seed混合。旧H在Hidden上的Raw−H约−0.08732至−0.08775，保留为错误实现控制，不能再代表正确物理中心的失效。旧160全部41对照每catalog保持描述性，没有偷加bootstrap主检验族。来源：legacy_contrast_summary.csv、legacy_seed_contrasts.csv。

**独立验算、重复执行与论文复核。** [独立数值验算](<E:/JINST REP/_research_extension_20261108/independent_information_numerical_review.json>)为PASS_COMPLETE_INFORMATION_NUMERICAL_REVIEW：80源validation均值、480个fresh model evaluations、每catalog全部2000 bootstrap draws及16/42/336族区间已独立重算；它证明算术和工件一致，不证明bootstrap假设或有限样本覆盖率。source analyzer另已实际重放全部validation预测，最大误差0。
[Full20 identity review](<E:/JINST REP/_research_extension_20261108/full20_rerun_identity_review.json>)确认20/20配对Full重新执行的state tensors以及source/C/D logscore bytes完全相同。240指注册运行身份，不能称240独立或全部数值不同的预测器；新Full20不提供独立重复科学证据。只应声称所查state/observed-output一致，不额外声称已证明所有可能输入上的函数相等。
主文当前SHA256 b073af9d3cb715ac40960be61e1a604f49521fbcaac74e6638a4a013e95557e7。已报告并由root修正Hidden5−Full5上界：0.019554应为0.019578；当前正文已正确。摘要、主16范围、局部8行计数、source56数值、Full20措辞和讨论的非因果/固定模型边界均与结果一致，未发现尚未关闭的必须科学更正。非阻塞建议：机制段可加入上面的一个PIT改善/恶化实例，使“诊断帮助解释”落到实际证据；不能据此进一步声称独立因果贡献。

**可支持的主张：** 相同名义架构下，物理目标中心相对Raw的收益随所测信息编码而非单调变化；新数据上固定模型平均收益为正且有可定位的局部平均损失；有限信息下中心能有价值，但收益与绝对预测质量不同。

**不能支持的主张：** 更少信息总使物理先验更有用；Coarse2普遍最佳/最精确；发现了information-compatibility threshold；负bin等于单事件伤害或整片区域必然失败；density四项就是location/scale/shape的因果分解；物理中心优于所有fitted baselines；事件区间覆盖未来重训/新物理环境；几何密度比百分比就是预测精度或下游收益。

输入绑定（本笔记未改这些文件）：

- [ANALYSIS_MANIFEST.json](<E:/JINST REP/_research_extension_20261108/fresh_analysis/C/ANALYSIS_MANIFEST.json>) — SHA256 `16da3e60e19112a442e5c4439624b0cfa309c37ddd23fffca5135d05c909e67b`
- [ANALYSIS_MANIFEST.json](<E:/JINST REP/_research_extension_20261108/fresh_analysis/D/ANALYSIS_MANIFEST.json>) — SHA256 `ec12e42f3933def209d4cb11e94cee92a254d69ab64ea3807f2229aaa04617a7`
- [independent_information_numerical_review.json](<E:/JINST REP/_research_extension_20261108/independent_information_numerical_review.json>) — SHA256 `03b29a14973d25a01a864bb8eb725e5ee89eaffe9f3c9dea3cef9c432624af3a`
- [full20_rerun_identity_review.json](<E:/JINST REP/_research_extension_20261108/full20_rerun_identity_review.json>) — SHA256 `39964c1183ba8b552ee0edbf4256c7ceec66d5e9dd2e6d21c6e20ca56bffe2b2`
- [source_location_diagnostics.json](<E:/JINST REP/_research_extension_20261108/source_analysis_retry1/source_location_diagnostics.json>) — SHA256 `a47a5280fea3236c9de0c604ca9e8a08d9f7a2e21cb59bcb38a95541e72ebe65`

补充只读核验：逐一读取新80模型在C/D的160个已保存log_support_cdf数组，所有元素均为float32的±0；这一观察限于保存精度、这些模型及事件，不证明数学上严格无support效应。100% Hidden5在C最低p-bin的base/Jacobian配对贡献为−0.0490219424/+0.1395499987，最高p-bin为−0.1147646161/+0.0850750983；D对应−0.0479627674/+0.1398179612与−0.1133027186/+0.0849262753。以上只读取已有分量，没有新增推理。源score_manifest分别位于fresh_scores/C与D，绑定全部原数组。

关于图与正文：source_location图展示的是position RMS（原单位面板换算MeV/c、另面板标准化），而本笔记及正文机制段表述的是variance数值；二者不可混称。root已修图caption，未改变统计输出。
