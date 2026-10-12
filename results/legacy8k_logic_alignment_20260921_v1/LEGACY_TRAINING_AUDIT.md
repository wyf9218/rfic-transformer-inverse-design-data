# 旧80.85%来源模型：训练与decoder恢复审计

审计时间：2026-09-21 19:39 UTC。仅读取原件、张量形状和保存的normalizer；未运行推理、训练、测试或EM。模型为 `real10k_center15ghz_seed20260711`，不是原100K模型。

| 项目 | 历史原件确认值 |
| --- | --- |
| source / gradient train / validation / sealed test | 10000 / 7871 / 1227 / 902 |
| Forward / inverse | 10→256→256→256→4 / 4→256→256→256→10 |
| 激活 | GELU，保存的推理实现为tanh近似 |
| Q标签 | 同频点min(Qp,Qs)；目标为对称exact回归 |
| 响应loss | declared-range平衡MSE，四量程[2.5,2.5,20,0.8] |
| 几何anchor / topology penalty | 0.01 / 0；旧模型没有启用topology penalty |
| decoder | 每维独立sigmoid到训练几何包络，raw logits clip [-40,40] |
| refinement | 0步，disabled |
| 训练seed / sampler seed | 模型20260711；F20260711、I20260712 |
| 学习率 / weight decay / batch | 0.001 / 1e-6 / 1024 |
| 实际预算 / 最佳checkpoint | F/I各1200 updates；best F1200、I1060 |
| split | 四维各4桶，整cell分组；train/val/test cell=83/13/11，重叠0 |
| 评价状态 | validation-only；test未访问；REVIEW_REQUIRED_VALIDATION_ONLY |

真实训练记录在[原summary](/Users/wyf/Documents/模拟变压器AI反向建模/reports/model_training_monday_20260720/multifrequency_real10k_pilot_20260720/model_seed_20260711/A_center_15ghz_4d/physical_feature_tandem_inverse_summary.json:25)及[原stage合同](/Users/wyf/Documents/模拟变压器AI反向建模/reports/model_training_monday_20260720/multifrequency_real10k_pilot_20260720/model_seed_20260711/A_center_15ghz_4d/stage_checkpoints/stage_checkpoint_contract.json:415)。F/I结构还由原NPZ各四层weight形状直接确认。原summary SHA为 `90a81532f7342ae7348248ea45889368fbb13ed2b6ee8f89e789f80f6811a3fa`；weights SHA为 `ffea66dfdd0bb252e402e1ade70f5e8768511e26f2978b1d73b1817a0221e42a`。

## 精确恢复的计算关系

设标准化响应残差为 `e=(response-target)/x_scale`。原响应MSE按 `mean(w*e²)` 定义，w=[1.1522627181686569,1.3089853003412328,0.6742038033806842,0.8645481781094257]；这与按原四量程归一化的对称MSE只差正的整体常数。该结论来自[原loss合同](/Users/wyf/Documents/模拟变压器AI反向建模/reports/model_training_monday_20260720/multifrequency_real10k_pilot_20260720/model_seed_20260711/A_center_15ghz_4d/physical_feature_tandem_inverse_summary.json:328)，不是由当前runtime的默认Q语义推测。不存在已记录的minimum-Q截断或+2 margin。min(Qp,Qs)是物理标量的提取定义，不能混同为minimum目标。

原response权重按warmup_ramp_adaptive_ema变化：optimizer-update域，warmup=.05、ramp=.25、EMA=.95、multiplier区间[.25,4]。旧trainer源码未恢复，因此不把新trainer中的具体schedule代码称为原始实现。

所有十组保存的normalization数组已逐项与原stage合同完全相等，包括x/y均值、尺度、feature上下界、geometry上下界及loss权重/量程。复用原数组即可，无需重新拟合。保存的几何上下界是标准化坐标，转换关系是：

`x_std=(x-x_mean)/x_scale`

`g_std=lower+(upper-lower)*sigmoid(clip(I(x_std),-40,40))`

`g_um=g_std*y_scale+y_mean; proxy=F(g_std)*x_scale+x_mean`

对应[既有portable runtime](/Users/wyf/Documents/模拟变压器AI反向建模/reports/seven_model_10k_v3_20260907T222800Z/BB00_reference_package_v1/runtime/frozen_mlp.py:149)。GELU在同文件198行起。没有耦合hard-feasible decoder，也没有几何修复；包络限制不是DRC证书。

原F最佳stage loss=.17569316433120366，与validation平衡RMSE相符；I最佳loss=.05544069751122155，与`validation平衡response RMSE + .01×geometry RMSE`相符。[F stage](/Users/wyf/Documents/模拟变压器AI反向建模/reports/model_training_monday_20260720/multifrequency_real10k_pilot_20260720/model_seed_20260711/A_center_15ghz_4d/stage_checkpoints/forward_proxy_stage.json:9)、[I stage](/Users/wyf/Documents/模拟变压器AI反向建模/reports/model_training_monday_20260720/multifrequency_real10k_pilot_20260720/model_seed_20260711/A_center_15ghz_4d/stage_checkpoints/tandem_inverse_stage.json:9)记录best更新1200/1060。最佳比较器的完整原代码仍属UNKNOWN。

## 唯一实质来源缺口

原summary和stage合同绑定trainer SHA `1288e8f5dbedb74e64fe77d044cb8ca67b24edde172b2f21e988dc0f93a9ef4a`。原命令路径现存文件及public sibling均为 `92988524b08b15a2388f655f6239070889098024e49ee184832f69876f7db3be`，不匹配。既有[BB00 replay收据](/Users/wyf/Documents/模拟变压器AI反向建模/reports/seven_model_10k_v3_20260907T222800Z/BB00_reference_package_v1/evidence/REPLAY_RECEIPT.json:244)已明确original trainer identity unproven。

因此：冻结模型/normalizer/推理decoder能够复用；原始训练逐步重现、optimizer具体版本与状态、初始化细节、schedule实现和源码比较器不能声称已完整恢复。不做全库搜索，也不依据当前默认值补历史缺项。

## 最小复用入口

1. 旧模型直接复用现成[BB00 portable package](/Users/wyf/Documents/模拟变压器AI反向建模/reports/seven_model_10k_v3_20260907T222800Z/BB00_reference_package_v1/PACKAGE_MANIFEST.json)的`load_baseline.py::load_package`（225–252行），或其runtime的`FrozenTandemMLP.load(..., contract_path=...)`与`predict`。冻结runtime SHA `5fe2ac6a7f3c3ed119b5350f9a96e3726c70bec147ce9e253edbd7fd5fe40d03`；不重建训练框架。
2. 原100K跑同一8K frame时保留它自己的weights/normalizer/decoder，只复用已闭合的外部Q扫描和评价规则。不同模型训练/decoder的结果只能作描述性系统对照。
3. 如后来明确要求按旧配方重训，应使用新版本输出并标“历史配方重建”，在原trainer缺口闭合前不称精确历史训练重现。本审计没有授权或启动重训。

全部路径、SHA、参数及UNKNOWN边界见同目录`LEGACY_TRAINING_AUDIT.json`。

