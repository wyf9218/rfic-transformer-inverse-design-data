# 模型结构、算法与设计选择：交接讲义

编写日期：2026-10-04 CDT；最后核对2026-10-05 UTC。本讲义只复用已有summary、结构清单、冻结源码及终态证据；没有重新训练、推理、评价或运行模拟器。适合配合本目录交接手册的55–100分钟代码讲解。

## 1. 先确定正在讲哪一个模型

下面三个对象都有15 GHz四物理指标输入和10维几何输出，但不是同一个模型；这是版本说明，不能据此把差异归因于“数据更多”或“某项优化”。

| 项目 | 原100K部署模型 | real10k模型 | 2026-09-21对称配方复刻 |
|---|---|---|---|
| 身份 | original100K，seed20260713 | real10k_center15ghz_seed20260711 | legacyrecipe_D100k_seed20260711 |
| source / 梯度train / validation / test行数 | 100000 / 78891 / 9740 / 11369 | 10000 / 7871 / 1227 / 902 | 100000 / 78891 / 9740 / 11369 |
| F正向网络 | 10→256→256→128→4 | 10→256→256→256→4 | 与real10k层宽相同 |
| I逆向网络 | 4→512→512→256→10 | 4→256→256→256→10 | 与real10k层宽相同 |
| F / I参数数，含bias | 102020 / 399114 | 135428 / 135434 | 135428 / 135434 |
| decoder | hard_feasible_topology_v1 | 独立sigmoid训练包络 | 独立sigmoid训练包络 |
| I响应loss | 加权相对MSE；Q为下限，训练加2 | 四指标对称、量程平衡MSE | 四指标对称、量程平衡MSE |
| geometry anchor / topology系数 | 0.01 / 0.02 | 0.01 / 0 | 0.01 / 0 |
| batch / F、I各自实际更新数 | 4096 / 3600 | 1024 / 1200 | 1024 / 1200 |
| 保存的F / I最佳更新 | 3560 / 3580 | 1200 / 1060 | 1200 / 1200 |
| 此处讨论的应用推理 | 一次I生成一候选；不扫描Q | Q=10…20共11候选后用F选取 | Q=10…20共11候选后用F选取 |

原100K参数数来自已核权重形状的[结构清单](/Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/paper_actual_structure_package_20260919_v1/network_structure.json)。另两列参数数按已记录层宽逐层计算，公式是Σ(in×out+out)，本轮未重新解包权重。real10k层宽有公开合同佐证；复刻层宽还经过原postprocess实际加载检查。原100K与real10k的历史summary采用validation_only，预留test行数本身不证明已经执行独立test评价。

## 2. 张量、层与normalizer

用`x`表示四个物理量，顺序固定为`[Lp(nH), Ls(nH), Q, |k|]`；`Q=min(Qp,Qs)`是标签的定义。**取两个线圈Q的较小值，并不自动表示“最低Q需求”**；后者由各版本loss合同决定。对用户展示统一写Q，旧字段中的`q_min`仍须保留身份。

`g`为10维几何，单位µm，严格顺序为：primary外宽、primary外高、secondary外宽、secondary外高、共享线宽、primary terminal span、secondary terminal span、offset、primary feed extension、secondary feed extension。历史trainer把几何变量命名为`y`，所以`x`不是正向网络的几何输入；读代码时最容易在这里弄反。

原100K逐层矩阵shape（batch行向量约定`z=a@W+b`）：

| 网络 | 四层W shape | 四层bias shape | 隐藏层/末层 |
|---|---|---|---|
| F | (10,256)、(256,256)、(256,128)、(128,4) | (256)、(256)、(128)、(4) | 三层GELU / 线性 |
| I | (4,512)、(512,512)、(512,256)、(256,10) | (512)、(512)、(256)、(10) | 三层GELU / 线性logits |

real10k和复刻的F中间两矩阵均为(256,256)，末层(256,4)；I首层(4,256)、中间两矩阵(256,256)、末层(256,10)。这里的“深度3”指三隐藏层，实际有四组可训练W/b。GELU使用tanh近似，不是ReLU；线性logits之后还有不含可训练参数的decoder。

normalizer仅由所属模型的train行拟合：`x̄=(x−x_mean)/x_scale`，`ḡ=(g−y_mean)/y_scale`，scale为总体标准差`np.std(ddof=0)`并以`1e-12`托底；几何上下界也只取train。F学习`ḡ→x̄`，I学习`x̄→logits→ḡ`，输出物理量用相应scale和mean还原。[冻结trainer `_normalize`](/Users/wyf/Documents/模拟变压器AI反向建模/reports/frozen_mlp_synthesis_toolkit_20260804/toolkit/rfic_transformer_inverse_synthesis_toolkit_v1/model/trainer_runtime.py:1773)

权重、列顺序、normalizer和decoder合同共同构成一个模型。不能给原100K权重换上real10k normalizer，也不能只换decoder后继续使用原模型名字。复刻在原100K的train上重新拟合normalizer，从随机权重重新训练，不是给原权重换一个Q公式。

## 3. Tandem训练到底更新什么

第一阶段用已有EM标签训练F：几何→四项响应。F的Q始终回归准确的储存标签，包括原100K模型；原100K的单边Q只在逆向目标语义开启时使用。选好F checkpoint后进入第二阶段。

第二阶段：`x̄ → I → decoder → ḡ → 已冻结F → 响应loss`。逆向几何通常不唯一，Tandem允许找到能产生相近响应的几何；0.01 geometry anchor又让结果适度靠近该训练行的已知几何。因此本项目这些模型并非完全“无几何监督”的纯响应loss。

“F冻结”是F权重不更新，不是切断梯度。NumPy实现先算loss对F输出的导数，经`_backward_input_gradient`得到loss对F输入几何的导数，再通过decoder导数/Jacobian回到I；最后只有I的W/b执行Adam。每次更新都不调用EMX。[F输入反传](/Users/wyf/Documents/模拟变压器AI反向建模/reports/frozen_mlp_synthesis_toolkit_20260804/toolkit/rfic_transformer_inverse_synthesis_toolkit_v1/model/trainer_runtime.py:3391)、[逆向训练完整梯度链](/Users/wyf/Documents/模拟变压器AI反向建模/reports/frozen_mlp_synthesis_toolkit_20260804/toolkit/rfic_transformer_inverse_synthesis_toolkit_v1/model/trainer_runtime.py:4179)

原100K总梯度目标：`λ(t)Lresponse + 0.01·mean((ḡ−ḡ_label)²) + 0.02Ltopology`。real10k/复刻的topology项为0。Adam学习率均0.001，weight decay为1e-6（在本NumPy实现加到更新权重的梯度上，不是AdamW式说明）。

`λ(t)`也不是永远等于1：warmup阶段响应权重0，随后线性ramp，再按响应/geometry loss的EMA比例调节，倍率限制[0.25,4]、EMA decay=0.95。原100K warmup180、ramp720更新；real10k配方warmup60、ramp300更新。warmup仍有geometry anchor；固定更新预算已完成也不等于已证明收敛。[schedule实现](/Users/wyf/Documents/模拟变压器AI反向建模/reports/frozen_mlp_synthesis_toolkit_20260804/toolkit/rfic_transformer_inverse_synthesis_toolkit_v1/model/trainer_runtime.py:4289)

## 4. 两种decoder解决的范围不同

独立sigmoid：每个logit先数值裁到[−40,40]，`ḡ_j=lower_j+(upper_j−lower_j)·sigmoid(logit_j)`。它把每一维留在训练包络内，但各维独立，不能单凭这一点保证terminal和外框、offset和feed之间相容。[独立投影](/Users/wyf/Documents/模拟变压器AI反向建模/reports/frozen_mlp_synthesis_toolkit_20260804/toolkit/rfic_transformer_inverse_synthesis_toolkit_v1/model/trainer_runtime.py:3409)

原100K硬耦合decoder先做包络映射，再按实际外宽/高限制两侧terminal span，并按共享线宽、offset、外宽和已保存的port-ground设置重新限定最短feed。terminal上限包含`min(train_upper,W,H,H−(√2−1)min(W,H))`；feed最小长度用耦合几何和保存的22µm offset/clearance和计算。它使用完整Jacobian反传，不能用逐坐标sigmoid导数代替。[硬decoder实现](/Users/wyf/Documents/模拟变压器AI反向建模/reports/frozen_mlp_synthesis_toolkit_20260804/toolkit/rfic_transformer_inverse_synthesis_toolkit_v1/model/trainer_runtime.py:3417)

这些是模型内的解析约束，decoder本身不做网格吸附、不生成GDS，也不能代替完整DRC或fresh EMX。历史SRF筛选按各旧实验记录解释，本交接不恢复后来已取消的半SRF硬门。改变decoder是改变方法，不是只改一个显示选项。

## 5. 三种“误差”必须分开讲

**训练loss。** 原100K对Lp/Ls/|k|用`e=prediction−target`，对Q用`eQ=min(predQ−targetQ−2,0)`；除以`max(abs(target),floor)`后平方加权，floor为[0.5,0.5,5,0.05]，原始权重[1,1,2,4]归一为[0.5,0.5,1,2]。Q超过目标加2以后没有该项短缺惩罚。+2是训练guardband，推理输入不额外加2。[loss实现](/Users/wyf/Documents/模拟变压器AI反向建模/reports/frozen_mlp_synthesis_toolkit_20260804/toolkit/rfic_transformer_inverse_synthesis_toolkit_v1/model/trainer_runtime.py:5105)

real10k/复刻对四项均用对称误差；量程`S=[2.5,2.5,20,0.8]`。代码在标准化空间使用`w=(x_scale/S)²/mean((x_scale/S)²)`，再算`mean(w·(x̄_pred−x̄_true)²)`。因此曲线中的feature-balanced normalized RMSE带有mean-one缩放，不应把其数值直接称为百分比准确率，或与下面未额外缩放的候选分数混为同一数字。

**选checkpoint。** 原100K I在validation上按物理cell达标率的P10、最低值、等cell均值、总体行达标率及最差特征MAE依次择优；此处Lp/Ls/|k|阈值10%，Q要求达到原目标，不附加训练+2。复刻的`training_objective`名称容易误解：实际在validation上算对称response RMSE+0.01 geometry RMSE；不是用test选模型。[验证选择实现](/Users/wyf/Documents/模拟变压器AI反向建模/reports/frozen_mlp_synthesis_toolkit_20260804/toolkit/rfic_transformer_inverse_synthesis_toolkit_v1/model/trainer_runtime.py:3963)

**应用选候选与最终报告。** 对称配方的Q扫描固定Lp/Ls/|k|，对整数Q=10…20生成11个候选，以各候选自己的目标Q计算`s=sqrt(mean(((F(g)−target)/S)²))`，取最小；精确同分取较低Q。该F分数只是代理排序，不能叫EM测量，也不是单纯选最大Q。[选候选代码](/Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap_paper_figures_overnight_20260921_v1/legacy10k_recipe_on_original100k_20260921_v1/postprocess.py:87)

选定候选后报告`r_j=abs(pred_j−target_j)/abs(target_j)`，联合误差为`max_j(r_j)`；在候选有限、几何包络和proxy有效性门通过的前提下，四项均≤10%才达标。这个相对误差分母是各自target；排序分母是固定量程，两者不同，不能用报告指标重新偷偷挑候选。近零|k|的相对误差敏感，要保留绝对误差/量程误差并说明适用范围。非有限候选/失败请求必须保留在原请求分母。[报告误差代码](/Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap_paper_figures_overnight_20260921_v1/legacy10k_recipe_on_original100k_20260921_v1/postprocess.py:174)

原100K单次推理没有这11选1步骤；F只对返回候选作诊断。历史最低Q协议、后来四指标对称重评、Q扫描三个结果应各自标明。原one-shot配对64的EM证据不能转嫁给新Q扫描几何。

## 6. 怎样讲“优化”，怎样加载正确模型

这些实现可称为设计选择：量程平衡用于处理单位/尺度差异；geometry anchor用于限制逆向解的漂移；Q margin用于预留最低Q余量；硬decoder用于表达维度间约束；cell尾部选择用于关注稀疏/较差区域；Q扫描是应用时的有限候选选择。这里解释设计意图，不宣称各项单独带来了已证实提升。架构、数据、split、预算、decoder和推理流程同时改变时，只能给描述性对比；因果结论须另有预先冻结的控制实验。

公开checkout的`FrozenTandemMLP.load(model_dir)`默认读取`real10k_model_contract.json`，不会根据目录名自动切成原100K。加载原100K应同时显式指定其目录和`deployed100k_model_contract.json`；旧toolkit的`FrozenInverseModel.load(bundle_dir)`又是另一接口，要使用同一bundle配套代码。本讲义只指认入口，没有执行加载。[当前public默认位置](/Users/wyf/Documents/模拟变压器AI反向建模/github_release/rfic-transformer-inverse-design/rfic_transformer_inverse_design/synthesis/frozen_mlp.py:71)

本轮已验证结构、记录的训练配方、summary身份及9/21复刻COMPLETE终态；2026-10-04 04:57 UTC的MARS观察也读回复刻COMPLETE。其README顶部“刚启动”是保留的旧时点，不能覆盖[完成收据](/Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap_paper_figures_overnight_20260921_v1/legacy10k_recipe_on_original100k_20260921_v1/received_report_20260921T2011Z/COMPLETE.json)。real10k原trainer的绑定源码字节`1288e8f5…`尚未恢复；9/21使用冻结trainer `fbfda202…`实现已记录配方，故称RECORDED_RECIPE_RECONSTRUCTION，不能说逐位精确重放。原100K最初1.2M候选池完整生成来源也未闭合，不能称原100K纯LHS。接手人账号、迁移后运行及模型新几何的物理验证仍需分别验收。

## 7. 建议阅读与现场验收顺序

1. 先从本目录只读数据练习指出一行的4物理列、10几何列和S4P引用；说清source与train行数。
2. 阅读对应summary的`arguments`、`split_audit`、`weights_npz_sha256`；在三列模型表中指出当前身份。
3. 按normalizer→`_forward_with_cache`→decoder→`_train_inverse`读冻结源码，手画loss到I的梯度路径，说明F为何仍能传梯度。
4. 用同一个目标手算“Q低于/高于目标”的最低Qloss和对称loss区别，再区分量程排序分数与报告相对误差。
5. 从已有结果CSV追溯到模型、候选与证据等级；之后才在接手环境做一例明确模型身份的小样本加载/推理。无需现场重训或等待新EMX。

## 8. 本讲义的主要来源与SHA-256

以下小文件本轮已计算SHA；权重SHA是summary/manifest和既有结构导出中的身份引用，本轮未重读权重。源码行号以这里列出的字节为准。

| 来源 | SHA-256 |
|---|---|
| [原100K结构清单](/Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/paper_actual_structure_package_20260919_v1/network_structure.json) | `088d9e930cc0f300e161736a13692f41312d34b44bb1d405d5dd2120db1287f5` |
| [冻结trainer](/Users/wyf/Documents/模拟变压器AI反向建模/reports/frozen_mlp_synthesis_toolkit_20260804/toolkit/rfic_transformer_inverse_synthesis_toolkit_v1/model/trainer_runtime.py) | `fbfda2023f862af2f0b185e41c4bf899f97cb79b0d0d763d20501c03525e8a4f` |
| [real10k summary](/Users/wyf/Documents/模拟变压器AI反向建模/reports/model_training_monday_20260720/multifrequency_real10k_pilot_20260720/model_seed_20260711/A_center_15ghz_4d/physical_feature_tandem_inverse_summary.json) | `90a81532f7342ae7348248ea45889368fbb13ed2b6ee8f89e789f80f6811a3fa` |
| [9/21复刻summary本地镜像](/Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap_paper_figures_overnight_20260921_v1/current100k_forward_evidence_20260921_v1/physical_feature_tandem_inverse_summary.json) | `dd2773ba3eba618d7628ac22ddb9b7d99ed55742c5786799a4eb2b54a1dae76b` |
| [9/21配方适配器](/Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap_paper_figures_overnight_20260921_v1/legacy10k_recipe_on_original100k_20260921_v1/train_legacy_recipe.py:56) | `78487afe59f870bb372cc12c337382e052b3a9b88f90bdf85907471b8dea9c50` |
| [9/21评价代码](/Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap_paper_figures_overnight_20260921_v1/legacy10k_recipe_on_original100k_20260921_v1/postprocess.py) | `f45e3eb69581b20e41453e72c70a249790547a718bb5a6ca873b2617a8e9a393` |
| [public loader当前字节](/Users/wyf/Documents/模拟变压器AI反向建模/github_release/rfic-transformer-inverse-design/rfic_transformer_inverse_design/synthesis/frozen_mlp.py) | `f223aa8c8f62288cfcec3e4449018c0171b35629afe5eca46cc0a1d727d2fbf9` |

权重身份：原100K `c61a93d56136ad21686863aae72984c49eb0d33ce766923c8ec2d03d9c3177dd`；real10k `ffea66dfdd0bb252e402e1ade70f5e8768511e26f2978b1d73b1817a0221e42a`；9/21复刻 `ddc3476184911007132728722ae5fb0a00d1183ee867af80cbc6ccaf9dfc805b`。复刻输出完整索引见同级接收目录的[SHA256SUMS](/Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap_paper_figures_overnight_20260921_v1/legacy10k_recipe_on_original100k_20260921_v1/received_report_20260921T2011Z/SHA256SUMS)。
