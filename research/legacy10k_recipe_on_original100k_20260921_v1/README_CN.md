# 旧10K训练配方在原100K数据上的复现

实际启动：2026-09-21T19:53:53Z，MARS PID355138/start_ticks57465753。19:54:15Z已真实完成forward120/1200更新，inverse尚未进入；尚未报告训练完成。启动、参数核验、进度已接收至 mars_receipts_20260921T1954Z/，后续以远端新增终态为准，不重复launch。

远端运行根：/volumes/research-localdata/ywang3652/eucap_legacy10k_recipe_on_original100k_20260921_v1。训练日志 run/console.log，轻量进度 run/PROGRESS.json，整体状态 RUN_STATE.json；最终模型位于 run/，自动评价位于 run/analysis/。

用户于2026-09-21明确要求“复刻当时的模型训练”。本版本按此补充要求复现整套已记录旧配方，不只是给原非对称100K网络改Q loss。

## 保留与改变

- 母集：已定位原100000行，SHA fe94d15b0d083330fe7a64d3e059d083a5aa32d73394c85fc4b881e9904e68a2。
- 原100K split不变：train78891 / validation9740 / test11369；不是旧10K的7871/1227/902，也不重新按seed抽split。
- 来源为已保存15GHz Lp/Ls/Q=min(Qp,Qs)/|k|真实标签，不生成新训练数据，不恢复额外半SRF筛选。
- F：10→256→256→256→4；I：4→256→256→256→10；GELU。
- decoder复现旧的独立sigmoid训练包络，topology penalty=0。它不提供当前hard-feasible decoder的耦合约束保证，更不等于DRC/EM合格；物理验收标准没有放宽。本轮不运行物理验证。
- 正反向均从头初始化，不复用旧F/I权重。先训练F，再固定其权重、保留输入梯度训练I。
- 四指标对称declared-range MSE，量程[2.5,2.5,20,.8]；Q exact、margin0，geometry anchor=.01。
- 恢复旧batch1024、row_uniform、seed20260711、learning rate .001、weight decay1e-6；各1200次真实optimizer更新，每20次validation；旧warmup/ramp/EMA schedule保持。
- 从新版train单独拟合并保存normalizer，不加载旧10K normalizer，不把新normalizer替换进历史权重。
- checkpoint按validation的对称response RMSE加.01 geometry RMSE选取。test及已观察8000/10000目标不用于训练或选checkpoint。

## 复现的准确边界

旧summary90a81532…及stage合同可确认配方；原trainer绑定1288e8f5…的源码字节未恢复。本版本复用已验证冻结trainer fbfda202…中的对应MSE、GELU、sigmoid与Adam实现，名称为 `RECORDED_RECIPE_RECONSTRUCTION`，不是逐步精确重放旧训练。

数据源、训练量及split与旧10K不同。固定1200更新不等于保证新100K已收敛，也不预设会达到旧80.85%。实际更新数、最佳checkpoint、曲线与停止原因按日志报告。

## 执行与自动接续

train_legacy_recipe.py使用现有dataset adapter、split、冻结runtime；仅新增旧配方参数适配和轻量更新后观察。运行于独立目录、单训练进程、4CPU线程、nice19、16GiB虚拟内存上限；固定1200+1200更新，6小时后不进入新阶段，健康当前阶段收尾。

LAUNCH_INTENT及flock防止重复提交。完成的F/I stage可显式resume；未实现中途Adam/RNG无损恢复，不作此承诺。训练失败留存，不自动重复重训。健康进程不依赖AI继续在线。

完成后普通程序自动加载冻结模型，按同一旧逻辑扫描已保存8000及10000目标：Q10..20、固定量程RMSE选最小、四项相对误差联合统计，输出候选/源表/热点图。共同11369原test仅作固定模型的已有EM标签评价。所有随机目标热点图均标SELF_PROXY，不挪用以前64个one-shot EM结果证明新候选。

实际配置见CONFIG_MARS.json，实际状态见MARS接收目录中的LAUNCH_RECEIPT、RUN_STATE、PROGRESS；代码存在不等于训练已启动或完成。

本版本只在私有环境保留；未修改生产、GUI、历史模型/结果或公共GitHub。
