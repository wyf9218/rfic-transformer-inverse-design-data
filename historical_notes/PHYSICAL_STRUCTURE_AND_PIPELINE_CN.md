# 物理结构与版图—仿真链交接说明

整理日期：2026-10-04；用途：导师与接手学生 hands-on 讲解。本文整理既有证据，没有新生成 GDS、训练、推理、DRC、EMX 或服务器操作。下文版本和状态属于明确标注的历史样本/收据，不据示意图反推物理版本，也不宣称全历史数据已符合最新合同。

## 1. 先打开哪一份结构

以[实际结构依据包 README][package]和[样本身份清单][identity]为入口。样本为原 100K 模型 `current_foundry_qmin_response_only_seed20260713` 的 one-shot case002，evaluation=`4d59a6b85c25a6e4`；不是已证明的 100K 训练表成员，也未证明与某张旧截图完全相同。

包内已有[全部金属][allmetal]、[M10/AP 原边][primary]、[M9 副边][secondary]、[M5 地框][ground]四幅同坐标视图。它们是对实际 GDS 的离线导出，适合教学标注；原 9 月 19 日 BUILD_REPORT 已记录视觉核对，本轮未重绘或重新做视觉 QA。

## 2. 十个几何字段：模型输出与测量基准

严格使用下列顺序，单位均为 μm。数值是 case002 未网格化的模型输出，显示保留三位小数；完整精度及尺寸标注端点见[geometry_parameters.csv][geometry]。它们不是训练 bounds 或全数据 min/max。

| 序号 | 字段 | 本例值 | 学生应如何解释 |
|---|---|---:|---|
| 1 | `primary_outer_width_um` | 310.672 | 原边名义绕组路径左右参考线间距；不含馈线、抽头或半线宽扩展 |
| 2 | `primary_outer_height_um` | 162.259 | 原边名义路径上下参考线间距 |
| 3 | `secondary_outer_width_um` | 358.513 | 副边名义路径宽度，中心位于 `offset_um` |
| 4 | `secondary_outer_height_um` | 163.178 | 副边名义路径高度 |
| 5 | `line_width_um` | 11.892 | 两绕组、馈线、竖线与抽头共用的一个线宽自由变量 |
| 6 | `primary_terminal_y_span_um` | 21.639 | 原边两个信号馈线中心线的 y 间距；不是金属边缘净距 |
| 7 | `secondary_terminal_y_span_um` | 22.659 | 副边两个信号馈线中心线的 y 间距 |
| 8 | `offset_um` | −25.841 | 副边中心相对原边中心的有符号 x 位移；本例向左 |
| 9 | `primary_feed_extension_um` | 290.855 | 原边左名义路径参考线至名义馈线端的长度 |
| 10 | `secondary_feed_extension_um` | 169.572 | 副边右名义路径参考线至名义馈线端的长度 |

原参数 JSON 与实际 GDS 是不同层次：本例 GDS 使用 0.005 μm 网格，馈线实际外端另有 10 μm 地框投影重叠延伸，不能将该延伸倒算成新的模型参数。配置中的竖线偏置 12 μm 是“名义绕组路径参考线→竖线近侧金属边”；本例实际金属净距为 6.050 μm。固定 `spacing_um=8` 也不是两竖线间距或层间厚度。上述值来自本例，不能推广为所有版本的固定规则。

## 3. 实际层、端口与中心抽头

以下连接来自[连接表][connections]、[层与端口映射][layerport]及样本 GDS 读回；仅列自有结构关系，不复制 PDK、工艺厚度、材料参数或规则文件。

| 对象 | 实际关系 |
|---|---|
| M10/AP | 原边绕组；P001/P002 从左侧引出；原边中点以同层短横桥连接右竖线 |
| M9 | 副边绕组；P003/P004 从右侧引出；副边中点以同层短横桥连接左竖线 |
| M5 | 实际开槽地框；不能仅据旧总 manifest 的泛化文字画成实心地框 |
| 右竖线接地 | P007/P008；经 M10→M9→M8→M7→M6→M5 的实际过孔堆栈接地 |
| 左竖线接地 | P005/P006；经 M9→M8→M7→M6→M5 接地 |
| RF 端口 | P001–P004 共四个独立信号端口，各配对应 `_G` 参考标签；参考定义不表示金属短接 |

“8 个物理端点”与“4-port S4P”同时成立：P005–P008 属于接地辅助端点，不是另四个独立 RF 激励。两竖线用于中心抽头接地；不能在俯视图中把它们解释成上下 RF 馈线直接短接。

M10/M9 的六处投影交叠中，只有右上/右下两处接地 landing 有连接过孔，另四处无 M9–M10 过孔。M9 图中的两个小 landing 是真实结构，不能因看似孤立而删掉。俯视投影相交、图中地标注圆圈、pin 标签中心和实际 stitch 位置分别有不同含义。

## 4. 原 GDS、旧交付 GDS与预览的身份

| 工件 | 正确用途与身份 |
|---|---|
| `sample_layout/original_emx_batch.gds` | 真正 EMX 输入，57,344 bytes；批次含三个案例与 assembly，EMX 选中的 cell 为 `TRANSFORMER_001_4d59a6b8` |
| 旧单例交付 GDS | cell=`TRANSFORMER`；raw SHA 与批次文件不同。既有核对证明本例 248 个多边形及 pin-label 多重集合一致，不能将两文件 SHA 互换 |
| `historical_streamout_preview.original.png` | 原预览文件的字节复制；图片外观不能独立证明 GDS/模型/物理合同身份 |
| `previews/A–D` | 从真正 EMX 选中 cell 导出的同坐标 PNG/SVG；是派生展示件，不是新仿真输入 |
| `.sanitized.json` | 原配置的脱敏字段子集，记录源 SHA；不能称为完整原配置或可直接运行的 PDK 配置 |

批次 raw GDS SHA：`69a3fbbc916b4d183cd53be6e3875656f46114c51abe3f777e731db66599e2a1`；旧单例 raw SHA：`3d58e690c60c88ef88e6181f81ee6577d81605947528445ab4adb239e555c932`。本轮仅核验包内现有 raw GDS 与小型来源文件哈希，没有重跑原多边形比较。

## 5. 结构与算法约束为什么演进

| 变化/问题 | 已有证据 | 交接时的解释和边界 |
|---|---|---|
| 端口语义 | [8 月基线第 3 节][baseline]记录早期 S8P 分支，后冻结 4-port、辅助端接地 | 数据读取必须绑定端口顺序、参考地和提取公式；“都有 8 个标签”不足以判断兼容 |
| 地框合同 | 同一基线第 4.3 节区分旧 rectangular 地框与 `foundry_connected_slotted_m5_ground_frame` | 相同十维参数仍可能生成不同物理版图；旧 S4P 不能直接移到新地框名下，也不能把原 100K 的所有样本一概归为同一地框 |
| 解析几何耦合 | [原训练代码摘录][decoder]将 terminal 上限与绕组宽高相连，将 feed 下限与两绕组、偏移、地框间隙等相连 | 各维在独立数值 bounds 内不保证整体可构造；hard-feasible decoder 是对应合同下的解析约束，不等于通过 Calibre/EMX |
| 网格与 feed 不足 | [开发 128 诊断][diagnosis]保留原 128 分母；其中 35 解析失败：原边 feed 不足 21、副边不足 14，网格化前后均失败 | 这是该开发批的保存证据；不能把它归因于训练数据量，也不能混入 case002 物理成绩 |
| post-snap 端点构造 | 同一诊断中 6 GDS 失败案例的 48 个端口有 9 个边缘/重叠不符，案例最大差 5 nm；独立 via landing 检查 6/6 PASS | 上游端口失败导致的 via `NOT_EVALUATED` 不等于 via 制造失败 |
| 共享端点修补 | [9 月 12 日修补收据][endpoint]：显式策略 `shared_port_edges_20260912_v1`；局部六例回放改变 9 个端口、7 个多边形，最大 5 nm，原默认仍 legacy | 该收据只证明局部实现/回放，native 验证当时 NOT_RUN；改过多边形即产生新物理候选，不能继承旧 EM 标签 |
| DOE006 端面误识别 | [独立诊断说明][doe006]：实际端面宽与 10 μm overlap 正确，连续网格取整使中心偏 −7.5 nm，旧 nominal matcher 漏匹配 | same-edge lineage 草案仅重绑对应端面元数据；该说明明确 NOT_INSTALLED/DRC 与 EMX NOT_RUN，不能写成所有端口问题已解决 |

9 月开发[几何合同][devcontract]明确 0.005 μm 只用于导出，未采用 straight-through estimator；其边界、对称物理损失及严格 SRF 条件属于该开发路线。不得把它默认为原 seed20260713 的训练合同。

后续实际原工件复核也显示不能仅按版本名判定兼容：[9 月 14 日分组收据][grouped]对 15,310 个原 GDS 分组，其中 7,648 判结构不兼容、7,662 仅网格通过但 frame/port/原 DRC 资格仍未闭合。这里引用已保存统计，不重新扫描；数目是该历史截面，不是当前全项目合格总数。

## 6. 接手学生必须能讲清的完整链

原 100K 路线：真实 EM 标签 → 仅 train 拟合归一化 → 训练 F → validation 选择并冻结 F → 训练 I 经 decoder 与冻结 F 回传输入梯度 → 冻结 I 给出一次几何 → 解析门 → Cadence GDS 与候选身份绑定 → Calibre → fresh 4-port EMX → 从同身份 S4P 提取物理量。详见[流程证据][workflow]。

原模型 I 阶段使用 Q 下限目标与 +2 训练裕量，并含几何锚定 0.01、拓扑项 0.02；F 参数不更新，但梯度经过 F 返回 I。原 one-shot 推理没有 Q 扫描或候选选优，+2 也不是推理预处理。后来的整数 Q 扫描与四项对称匹配须作为另一推理/训练配置讲解。用户显示统一写 Q，底层历史定义仍是 `min(Qp,Qs)`。

验收口述：从十维字段定位本例两绕组、四信号端、两抽头及四接地点；指出一个投影交叉为什么不导通；用 candidate→GDS cell/hash→DRC→EMX command→S4P 串起身份；说明 proxy 误差、结构可构造、DRC 通过、EMX 完成、目标命中是不同结论。

## 7. 已知缺项与核验索引

历史布局生成器的精确源码字节版本、上游 120 万候选池的原生成器/seed、本例与旧截图同一性仍未闭合；不以当前 checkout 或图片补证。该包不是全依赖训练发行包，不含模型大权重或全部数据，也不代表原三案例规格全部命中。完整来源哈希见[包内 SHA256SUMS][sha]与[BUILD_REPORT][build]。

本轮已核验的小来源 SHA-256：README=`47aefd5328aa0c1f62a3a053bec9c1e65e969d84c54d4b6d19ee1ee93076aa6e`；几何表=`e2ad9e7cfa5b171ff86544a9ec03ffbdc5678d4382d616cd35eb31d4e068e7ce`；连接表=`770be345f0130575700970cb6fd8b1a6e0dc74e26d4498dc37947e14e7a0749e`；样本身份=`78a2bbf4ac35650ce155488e1f65c292d74be017fd2186b494f620eae9865556`；层端口映射=`933cdf00f65694acc049bcfbbaaa0c256ed7599e72c7eb8501be8403c000b41d`。

变化证据 SHA：开发诊断 SUMMARY=`defa097bc5ba4d62918efbcb48569eca39d1def36b99dbe885cd4675f383b904`；端点修补 RECEIPT=`b0c4d50e817d03bf540aa56cfab89cf5533da6756f4e47538236b542411e4dc5`；DOE006 README=`b378178fb8a8ba72ed87098c7f1ab08fa35ad4ec4f230d17f6ea10709911a3fa`；9 月开发几何合同=`09aea00c966b9e6d41d423b60c82e45ef6604533493e95afce071e02ba250447`；分组结果=`7b3289237a9d1a8ee444983c33dc3fb87a35edec4b491a701e4a620d5fa2209f`。

[package]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/paper_actual_structure_package_20260919_v1/README.md
[identity]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/paper_actual_structure_package_20260919_v1/sample_layout/SAMPLE_IDENTITY.json
[geometry]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/paper_actual_structure_package_20260919_v1/geometry_parameters.csv
[connections]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/paper_actual_structure_package_20260919_v1/layout_connections.csv
[layerport]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/paper_actual_structure_package_20260919_v1/sample_layout/layer_port_map.json
[allmetal]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/paper_actual_structure_package_20260919_v1/previews/A_all_relevant_metals.png
[primary]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/paper_actual_structure_package_20260919_v1/previews/B_M10_AP_primary.png
[secondary]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/paper_actual_structure_package_20260919_v1/previews/C_M9_secondary.png
[ground]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/paper_actual_structure_package_20260919_v1/previews/D_M5_ground.png
[decoder]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/paper_actual_structure_package_20260919_v1/source_code/network_trainer_exact_excerpts.py
[workflow]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/paper_actual_structure_package_20260919_v1/workflow_evidence.md
[baseline]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/chat_memory_20260820/SIMULATED_TRANSFORMER_NN_ENGINEERING_MEMORY_CN.md
[diagnosis]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/development128_failure_diagnosis_20260910_v1/SUMMARY.json
[endpoint]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/endpoint_fix_20260912_v1/RECEIPT.json
[doe006]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/batch_qualification_resume_20260912_v1/production256_doe006_endpoint_diagnosis_v1/README.md
[devcontract]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/production256_successor_20260912_v1/run_v1/GEOMETRY_CONTRACT.json
[grouped]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/grouped_reuse_20260914_v1/WORK_PACKAGE_RESULT.json
[sha]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/paper_actual_structure_package_20260919_v1/SHA256SUMS
[build]: /Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/paper_actual_structure_package_20260919_v1/BUILD_REPORT.json
