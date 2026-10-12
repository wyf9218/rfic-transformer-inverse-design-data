# 当前实际交付（2026-09-21 15:42 UTC；下方旧观察保留）

八组已接收重复的真实学习曲线现已全部导出并目视可读。最新Large17/29及D50k29分别为 `paper_curves/seed17/paper/Large/5e9777d3cb866d23/`、`seed29/paper/Large/4a5ef4309a719ea1/`、`seed29/paper/D50k/f2033b4e8fb71e22/`（均相对 `../repeats_20260921_v1/`）。下面15:38的“正在导出”现已闭合。新增统计仍只使用实际逐组保存结果，未齐family不生成均值/SD或新ZIP。四组Small/D10仍在原MARS队列运行。

真实追加重复已完成8/12，全部已接收并校验模型及逐组5项评价CSV；仍由原queue3720718接续。本轮新增D50k17、D100k29、Large17、Large29、D50k29，均F/I3600和原固定评价协议；SELF/COMMON分别1283/961、1283/1067、1139/919、1366/1117、1102/913，均10000请求、失败0。不是fresh-EM结论，未齐family不画均值/SD。D50k17和D100k29学习曲线已导出并目视；新Large两组及D50k29曲线正在增量导出，不重做旧五组。15:38现场Small17/29和D10k17/29四进程RNl，其余新重复均终态。实际结果路径 `../repeats_20260921_v1/received/`，新曲线路径 `../repeats_20260921_v1/paper_curves/`。原主图包/64结果不变，无新原生仿真或训练重启。

14:08:26只读实查：原队列3720718及5个训练子进程仍活，追加重复3/12完成、其余5运行/4排队；没有新第四组终态，未重启任何任务。补充包与CSV离线复绘入口已实交 `../repeats_20260921_v1/supplement_delivery/`，3项针对性入口检查通过；完整聚合、真实复绘与新ZIP尚未完成，必须等真实结果齐备。现有primaryZIP和全部原生结果不变。

D100k seed17新增真实终态13:23:36、评价13:23:48，队列已完成3/12追加重复。新训练非对称F/I各3600、best3580/3220；共同11369真实已有EM-label test、T10000 SELF1348/COMMON1095/fail0。完整run及五CSV本机接收校验通过，eval COMPLETE SHA07480faa0af4c895bbcf86b9c5ccced25472d0298c7dbf2633bef27a431de0cf。实际新曲线 `../repeats_20260921_v1/paper_curves/seed17/paper/D100k/a18e3714861f1827/`（COMPLETE96701b5b0fee0d9b0fb3d7b3244b429a444b2a96e27a33260a573e1776cdff0a）三图已目视。无新增EM或原模型替换。其余普通队列继续，未齐重复不汇总。主图ZIP保持不变。

新增两组真实重复已闭合：Medium seed17/29，各F/I 3600更新，13:05自动完成共同11369 EM-label holdout与同T10000评价，接收文件身份及5项评价CSV SHA分别通过。SELF命中1267/1332，COMMON命中1013/1069，分母均10000、请求失败0；代理不是新几何EM真值。原validation-only质量限定未改。已实交新曲线 `../repeats_20260921_v1/paper_curves/seed17/paper/Medium/7641bad43c5955cc/` 与 `seed29/paper/Medium/5999ebbaab9e4115/`；最佳更新3580/3580及3560/2000，六图目视可读，未重复primary曲线或推理。13:18:42原queue状态RUNNING、2/12重复COMPLETE；其余由原队列继续，未齐family不汇总均值/SD。完整首seed及最终64 ZIP仍是下方已交版本，新重复暂不在其中。

**主图总包已实际交付**：`/Users/wyf/Downloads/EuCAP_Primary_Comparisons_and_Paired64_20260921.zip`，191文件、7,203,841字节，CRC和归档逐文件SHA通过；ZIP SHA `f293df1451557030a2e764c2807d4efc7ea84f1d77e7b7e9bead957efc0221d4` 已主侧读回。覆盖两条首seed比较、六组学习曲线/成本、T10000热点、原100K分布、独立动机/流程及最终64配对图与误差表。只整理明确清单，没有重跑图、推理、训练或EM；追加17/29仍由3720718队列运行，本包不冒称多seed已完成。

最终64误差摘要 `../paper_paired64_v1/final64_error_summary_v1/`：14行MAE/RMSE/P95表、256行有符号残差、唯一proxy-only案例及英文结果段。EM-target四MAE为0.354551nH/0.352081nH/2.684/0.0928866；EM-proxy为0.0795791nH/0.0619655nH/0.441528/0.0199097。Q对称残差与minimum-Q短缺单列，未改变判据。唯一不同分类是paper_u20260921_05172：Lp、Qmin、K三项从proxy通过变为EM不通过。8/64=12.5%是实际联合成功率，63/64二元一致不等于98.44%逆向准确率。SUMMARY SHA c4cba0bdd54c6a4dc0da24543d57f699816d1a3ffe4210caecb729c94ef08f9f。

原64配对fresh EMX已完整结束：owner12:29:40、receiver12:30:00真实终态，12:30:23两进程已退出。全部64身份绑定结果已接收并核验128原JSON及8分析文件，物理链失败/pending/证据未闭合均0；proxy9/64、EMX8/64，8双方命中、1仅proxy、55双方未命中。最终三图SVG/PNG、CSV和英文图注已交 `../paper_paired64_v1/snapshot_0063/`，FIGURE_RECEIPT SHA97bfb9f17e33b8973f5e13c412a9b319ae1bcede6f114dbd5f7813a74c4e8dca；对应源 `../paired64_20260921_v1/received_snapshots/snapshot_0063/analysis/`。这不是64全部满足目标，也不是T10000全量EM验证；不能给六个模型作物理排名。首seed两family与原64现均具备可交付图源，正在一次性整理主图包。

追加seed17/29的12组训练队列3720718继续，无新提交或重启；12:10:51的Medium17/29和D100k17已完成F进入I720，Large17/29仍F2160/2280。无多seed均值/误差条可用，Goal仍ACTIVE。

11:39:26轻量现场确认：5组追加训练与原生owner/receiver均实存，最新保存的正向更新为Large17/29各720、Medium17/29各1800、D100k17为2160，其他7组排队。原64接收程序11:39:00发布53份镜像结果、身份拒绝0；此cut未完整本机接收，不提前解释物理合格/命中数。实际等待使用原session72921，无重复提交；普通程序继续运行，不需要聊天推动。首轮图已交，新增重复与最终64尚未完成。

新增真实运行：seed17/29的12组从头F/I重复已在MARS独立目录 `eucap_paper_repeats_20260921_v1/runtime_v1` 启动，queue PID3720718；5组运行、7组排队，最多20研究CPU线程/nice19，24h不再派发新阶段、在飞阶段正常保存。11:20:08读取真实进度：Large17/29最近已保存更新各1；Medium17/29及D100k17各120。不是首seed重训，不改子集/预算/评分，不把追加D100k当旧权重复用。实际receipt见 `../repeats_20260921_v1/mars_launch_receipts/`，config SHA5489b89a483672d7da2ec946c4de22ca99958680143e06116d892c850f1fac5e，launch SHA6df1563a9e81c4e100c44d426e1b8ec6cba565efea6fe8c6f46cc258d7967588。完成后现有程序一次评价与多seed聚合；此刻无重复实验均值/SD成果。

原配对64 owner3470666/receiver3481284仍正常接续，11:17:59远端已发布snapshot_0048，mirrored_result_count=49、identity_rejections=0；未完整接收本cut，不提前当作49个物理有效或宣称命中数。当前本机已交纸幅图仍为31例cut；待原64实际终态后导出最终配对图和主成果总包。三规模下载包及全部首seed图已可使用，不等待重复训练。

首固定seed20260713的两条训练比较现均完成：Large于10:54:42正常训练结束，F/I各3600、最佳3560/2340，10:55:23完成一次共同11369测试与同T10000评价。主侧已接收完整run/eval/curve并验证绑定SHA。原只读等待40902正常返回；eval、curve、paper辅助程序按原流程完成，没有重复启动。

Small/Medium/Large 的 SELF 命中为1369/1120/1343，COMMON为1050/1019/1015，均N10000、请求失败0；不能声称越宽越好，也不能称物理排名。四项正向MAE/RMSE随本次宽度增大下降，不代表逆向共同命中同步改善。实际宽度四组论文图位于 `../paper_publication_v2_env_recovery/architecture/a2203dd8a6a82398/`，COMPLETE SHA `f4da1c77398dd9a16a6e6b1f6bc77ab85dc089311d381a5888987efedf06802a`；13源文件绑定和4PNG目视通过。曲线 `../paper_training_curves_v1/Large/1ce3cabb6d346288/`，成本 `../training_cost_comparison_v1/width_only/6dd60cd90b838ec6/`。

配对64仍由原owner运行；最近远端cut10:37:58为40verified、24pending、5both-hit/1proxy-only/0EM-only/34both-miss、0chainfail；该cut未完整本机接收，当前已交物理纸幅图明确仍为31例。不得继续把31例时全部二元一致概括为当前结果。

补重复：遵照首seed后再补重复要求，资源实测load23.52/192CPU、可用470485MiB、盘1.8T，正在独立目录准备固定追加seed17和29（按用户给定序列前两项；不依结果挑选），原首seed全部保留。尚未启动则不报告训练中。追加版本不改变架构、预算、子集、共同测试/目标、Q语义或commonF；每个新D100从头训练，不复用原预测捷径。最终总包入口已实际inventory125文件、9163131B，拒绝31/64 partial；尚未打总包，不把准备当交付。

**10:04核验（图件10:01:10生成）**：09:59现场Large原PID仍RNl，最新09:57反向2040/3600。配对物理cut09:58:58已实收31/64，5双方均命中、26双方均未命中、0不一致/已记录链失败、33未验证；新增11，62原JSON+8分析文件SHA通过。最新实际物理纸幅图 `../paper_paired64_v1/snapshot_0030/`（三图SVG/PNG+源表；FIGURE_RECEIPT SHAa57df7b05b365d2598e68e32dbd69834fb3885a44bb965661b13354103cda8a2）已目视；不是10000目标全量EM结论。原14例图保留、不冒称31。源结果`../paired64_20260921_v1/received_snapshots/snapshot_0030/analysis/`。Large训练与原64自动执行继续，没有重启或新增任务。

09:38离线交付缺口已补：`../dataset_size_811_20260921_v1/analysis/publication_from_csv.py`（SHA5161457a8ac10d344ae2b1b248e000539b329822297527462fe82b184ab4142e）可只用随包3聚合CSV及2收据复用冻结draw，不再要求私有MARS路径。一次真实复绘和一次tamper拒绝通过，说明与命令在`../reproduction_check/README.md`。原ZIP未改，adapter作为独立补充；不新增模型评价、统计或EMX。

09:32新增交付：已完成三规模专包 `/Users/wyf/Downloads/EuCAP_DatasetSize_Completed_seed20260713_20260921.zip`，63文件/2524585B，CRC和逐文件SHA PASS，ZIP SHA`b478873c2bc68a39b8ca23f538719f16da09c6243716f70b3f7a7daceaab3678`已主侧读回。仅三规模真实图表/源表/曲线/成本/相关脚本，不含权重、原100K或未完成Large/64。清单位于`../dataset_size_delivery_v1/`。09:22:48 Large原PID仍RNl，最新09:21:49 I1080/3600；只读被动等待session40902，不重启训练。后续最新已收物理数值仍用下方20例cut，22例event待接收验证。

**三规模训练与比较已完成第一固定seed；宽度Large和原64物理任务继续。** 不是全部总目标完成。下面“历史cut”不代表当前进程状态。

| 组 | 当前真实状态 | F/I更新；最佳F/I | SELF / COMMON 命中（同10000目标） |
|---|---|---|---|
| D10k | 09:04:53训练完成；09:05:51评价完成 | 3600/3600；3320/3500 | 984 / 624 |
| D50k | 09:08:13训练完成；09:09:08评价完成 | 3600/3600；3580/1440 | 1195 / 1152 |
| D100k | 原seed13精确合同复用；评价完成 | 3600/3600；3560/3580 | 1420 / 1420 |
| Small | 新训练及评价完成 | 3600/3600；3560/3520 | 1369 / 1050 |
| Medium | 09:04:55训练完成；09:05:59评价完成 | 3600/3600；3560/1740 | 1120 / 1019 |
| Large | 09:14:36 PID3457208实际RNl | F3600；最新09:13:06 I840/3600 | 未评价 |

数据量与宽度两实验不混排名。所有新模型validation选best；原训练summary质量字段`REVIEW_REQUIRED_VALIDATION_ONLY`保留，完成训练不等于物理成功认证。共同11369原test标签预测都有限；T10000 Q为minimum、Lp/Ls/K相对误差≤10%，非Q扫描，两类proxy均不是EMX。D10/D50 reserved test仅1000/5000，但共同11369全部来自原test、与所有train/val隔离。

**核心数据量论文图**已于09:13:42普通程序自动发布，13源/图文件SHA传回核验、4PNG主侧目视PASS：`../paper_publication_v2_env_recovery/dataset/3b286727b01b8e5a/`，COMPLETE SHA`747bec90470cacbd62a351f5b260b01c692dcc0ea9d803450b080d2e38655699`。含四指标MAE/RMSE、实际Dataset size横轴命中曲线、两类共同网格热点图、CSV/英文图注/解读。四MAE随本次规模下降，但Lp/K的P95在D50k优于D100k，未掩盖非单调尾部。common scorer就是D100k自身F，不作独立物理裁判。无新推理仅为作图。

新真实权重和日志：`../dataset_size_811_20260921_v1/trained_models/D10k/`、`D50k/`；weights SHA分别`d01de3595defe5a2783b1d0a2a1b86e5c68b6c15a61a1fc15c5d5599b89c0886`、`c2bd6202351114c29bcf62864b7075d30d11f58587c84749980d3a2f33970a1b`。summary、history、validation/test输出及8+8评价图源已核验。Medium本机 `../architecture_100k_20260921_v1/trained_models/Medium/`，weights `d02ffc42cd75c184e1ec063f60440387a410df0abe2650f7880a7738ffb88352`；未覆盖原D100k。

三规模成本真实表：`../training_cost_comparison_v1/dataset_total/1f615d1005fa69d1/comparison_cost.csv` SHA`03ccd0be876f4cd9842d43f3a09370e1351a3ec997db40a7177d3af2e7d05d48`；D10k整体6060.70411s，D50k6260.28591s，分阶段秒数UNKNOWN；D100k历史耗时UNKNOWN，不写加速倍数。纸幅学习曲线：`../paper_training_curves_v1/D10k/fa437b7e4cc1ea90/`、`D50k/b7aa88424e402b6f/`、`Medium/63ac1619fecf38fa/`已导出并目视，不重绘旧Small/D100k。

**配对物理新cut09:11:58**：`../paired64_20260921_v1/received_snapshots/snapshot_0019/analysis/`，20已验证（较14例cut新增6），3both-hit/17both-miss/0discordant/44尚未验证/0chainfail。40输入JSON+8分析工件pins通过；ANALYSIS_RECEIPT SHA`d749df29d97b1d902820eb0af06692758fb1a0000d1552ed26c7a292983c2e75`。未验证不算失败，不把这20例写成10000整体物理精度；原候选/预算/唯一owner不变。当前已排版物理图仍为明确标注14例的旧cut，未改标题冒充20。

下一合法入口：Large原进程→既有评价/曲线follower→现有publication v2自动出完整width图；接收64的新增/最终实际终态。不重启已完成D10/D50/Small/Medium，不重复训练或推理。唯一owner3470666、接收3481284、评价3486449、曲线3509894、排版3571358于09:14:36现场存在。新数据量图已实际由v2运行发布，不再是仅“配置完成”。

## 历史cut（以下全部保留作为进展记录；最新状态以本页上方为准）

### 2026-09-21 09:04 UTC

当前目标 ACTIVE。原数据、模型、split、T10k和失败证据保持。不是“全部训练完成”。

当前论文自动排版PID为**3571358**，远端`DROOT/paper_publication_v2_env_recovery/`，09:03:40启动、09:04:39实际S；不是下方旧3552775。旧helper把venv路径resolve到缺少绘图库的系统Python，已在两family attempts0、无child的已证故障状态精确退役，仅辅助程序收到信号；训练/评价/native没有中断。v2真实依赖检查PASS（Python3.12.14/NumPy2.5.0/Matplotlib3.11.0），保留原截止1790066814.4370768。启动SHA b9b2226894fbb47431ac760bfff5d9842a6b4f695f66b4eeb9d1f8961efa5f54；本机`../dataset_size_811_20260921_v1/mars_receipts/publication_wait_v2/`。

最新训练cut09:02:10：四原训练PID仍存在，Medium已记录I3360/3600（09:00:44），Large I480/3600（08:59:41）。D10k/D50k仍在I，无逐步observer，不猜更新数。参数/实际更新/成本表合并入口已完成，真实D100k和Small源读取通过；完整family未齐，不填造横向CSV。

新增实际图件：`../paper_paired64_v1/snapshot_0013_v2/`的三类配对14/64图已成功导出并目视通过，刻度误报已最小修复，初版FAIL保留；FIGURE_RECEIPT SHA00cbeabea766ee0496ddeb74bfa8a74141d482648e6504c565ee0288d6705795。另`../paper_training_curves_v1/Small/ff388bd09c245e6c/`和`D100k/fce19a530ac57aeb/`已交纸幅F/I曲线及成本CSV，均复用保存曲线，不重新训练或评价。

最新增量：Large已完成F，最新08:45:42进入I120/3600；Medium I2520/3600（08:45:40）。D10/D50原PID继续，无完整终态。配对EM最新接收cut08:43:58为14/64 VERIFIED：2both-hit、12both-miss、0不一致，50未验证。`../paired64_20260921_v1/received_snapshots/snapshot_0013/analysis/ANALYSIS_RECEIPT.json` SHA195dda8ab66ea19655f27580960a2bde8f535237f1c3a3ab86de487f85e0cdbf；28真实输入JSON与8analysis图源均已核验。新14例紙幅图正在修轴外自动tick判定，初版失败保留，不影响EM或训练。

论文最终排版自动接续于08:46:54真实启动PID3552775，MARS `DROOT/paper_publication_v1/`。每300秒只读已有评价状态，family三臂及其汇总表发布后各执行一次，24h/1线程/nice19，不重复推理，失败不重试；当前两family均WAITING、attempts0。启动回执本机 `../dataset_size_811_20260921_v1/mars_receipts/publication_wait_v1/LAUNCH_RECEIPT.json` SHA3bae3e122f35376fa545196571f3f842630366058338a343e018206dff0d1c40；不要再次启动。完整比较图实际出现在remote `paper_publication_v1/figures/<family>/<identity>/`后再做最终目视。

最新现场08:38:04 UTC：D10k、D50k、Medium、Large原PID均存在；Medium最新记录反向2040/3600（08:37:29），Large正向3360/3600（08:36:41）。D10k/D50k已完成F各3600，I继续但无逐步计数入口。Small已正常完成，不重启。普通评价和曲线接续均已处理D100k与Small，其余等待真实训练终态。

最新配对物理结果cut08:30:58：11/64已验证，proxy/EM联合均命中2、均未命中9，两类不一致各0；53未验证，不等于失败或均在飞。相比5例cut新增6。`../paired64_20260921_v1/received_snapshots/snapshot_0010/analysis/ANALYSIS_RECEIPT.json` SHA dbbb87f7ae9d20e20df16fbd21e53b06d945dd288081a5e86eed445e477e858f；source-index SHA b38bc97258ec3c236269489676317f7dbc1c2dc690d608413353e967b25d744a。22输入JSON及4图已传输核验；不能从11例宣称T10000物理覆盖。

新增正式纸幅图源：`../paper_distribution_v1/`（COMPLETE SHA2d6a759aaf077ef167623e73f61492ccb90254e68bf784ae5e1e65f326aef7c1）与`../paper_methods_v1/`（DELIVERY_CHECK SHA028fe7b04da98c306eeec9d78b311e5935572e1adc8432c5759443213915fee9）。分布复用已有100K分箱CSV，流程严格区分原非对称F/I、冻结梯度、one-shot和物理验证。三来源proposal与五配额acquisition已分开说明；Fig1引用原已验收文件，不重绘。均已检查PNG。

最新：Small已经完成，不再报告5模型都运行。08:17:01 F/I各3600终态COMPLETE，08:17:43普通评价自动完成；共同11369已有EM标签test，T10000 SELF_PROXY1369命中(13.69%)、COMMON_PROXY1050命中(10.50%)，失败0。forward MAE Lp0.0984008872nH、Ls0.0693068703nH、Qmin0.350954035、|k|0.0155201019。其他D10k/D50k/Medium/Large训练继续。不能依据Small和历史非对称D100k两点宣称三宽度排名。

Small模型本机：`../architecture_100k_20260921_v1/trained_models/Small/`，weight SHAab641c846cc03653bc308e73166b3383960c695e44d8da86bc426a18d0d2d03d；结果：`../dataset_size_811_20260921_v1/analysis_received/Small/`，receipt SHA5fe14c12ae2166f5a65d05739d117669bf3b93f3394616253e8c2a4dadd35539。曲线/成本：`../dataset_size_811_20260921_v1/training_curves_cost_v1/Small/7f18caf9db438301/`，receipt SHAc9a0cd1f5e1dcf747449998ccf84571d652f3769c5e950b3c6984f101fb7b4db；F925.4103s/I1729.4390s、whole2659.99732s，参数34948/34954。完成run、8评价图源和8曲线图源已传回并核验；不重复推理。英文图注随结果保存。

08:14补齐普通曲线/成本接续：只读导出程序PID3509894/start53268246已在MARS实际启动，读取5个既有模型的完成终态，不推理、不训练、不碰原生owner；60秒检查、24小时上限、每臂只导出一次、失败不自动重试。入口DROOT/analysis/launch_curve_exports_once.py，manifest SHA1c2602e65fcc616441f06618a6646a8cef45247d7241db66058bce42ac74ab88，exporter SHAa9f6e71461d6b5f0f4481cebbd94029aa029f94ffcf29698987a9305ffe4eedc，启动收据SHA7d71bd50e9200d0d3a72cdfbcb68e6a90d6037b8566ae1e7bb62607cab39595d已传回mars_receipts。未修改已经运行的训练或评价脚本。

原D100k真实学习曲线已交于 ../dataset_size_811_20260921_v1/training_curves_cost_v1/D100k/16ff31386c8bba65/，含learning_curves与training_cost的PNG/SVG、逐事件CSV、成本CSV、英文图注；COMPLETE SHA23fd53cc151bb378fb0592fbb179be2063c98b4c2095ec24815d7adf35ae3c81。F/I各180个保存验证事件、各3600更新、各14200380真实行抽取，best更新3560/3580；两图主侧已目视。历史墙钟UNKNOWN，未补0或猜耗时。原模型共同test英文图注 CAPTIONS_AND_READING.md SHA c874e260859990633459a808f9c61c232fdfbdc13496e5ec2c85f5f285426d90；不把每模型normalized训练曲线当跨数据量物理精度。

物理增量cut08:04:58：已接收5个身份闭合的配对fresh EM（相比上一cut新增3），联合proxy/EM均命中1、均未命中4、两类不一致各0；59未验证，不写成59现场pending或总体失败。新目录 ../paired64_20260921_v1/received_snapshots/snapshot_0004/analysis/；ANALYSIS_RECEIPT SHAdcedeee162d71708a1d072251b7e8356524d844df6d69d5edec7f252260c7758，source-index SHA68ae15c33bcde30c21353419c66809752d36d83521066210ebb38c2f40dfd758。传回10 terminal/feature JSON、4图、index和receipt SHA全通过，Q–K图已目视；无配对证据格为灰色，未插值。旧snapshot_0001保持不变。

08:05增量：五个训练PID均实际存在、状态R。Small已进入反向并记录2160/3600更新；Medium已进入反向，最新记录1/3600；Large正向1680/3600。记录各自时间，不把最后保存更新当作精确当前更新。D10k/D50k正向各3600已闭合，反向运行但没有逐更新计数入口。没有新增训练或重启。

07:58已部署普通完成后评价程序，PID3486449/start53173431，源位于数据量远端root的analysis/，输出analysis_outputs/；每60秒只读6模型终态，24小时上限，每组只评价一次，失败独立留证不自动重试。完成后自动导出共同test、T10k、跨模型比较CSV/图像；不控制训练或模拟器。COMPLETION_CONFIG SHA fd028be19e49e9c8adf1a6faec12f1e691ceb3fda25d66f60ff3e7d2f071efa5。

原D100k共同11369留出EM标签评价已于07:58:34完成，全部有限、失败0：Lp MAE0.0776500162 nH、Ls MAE0.0515458822 nH、Qmin MAE0.282031862、|k| MAE0.0117340844。这是真实已有EM标签留出评价，不是新几何fresh EM。结果位于 ../dataset_size_811_20260921_v1/analysis_received/D100k/，包含逐样本预测、MAE/RMSE/P95表、散点图及T10k热点图；COMPLETE SHA77753749147e63148b8d7b680b9c557d0beb15decf17ce909ac5eb6528ec12dc。传回8文件SHA均通过，正向散点PNG已目视。T10k仍复用原保存预测，0次新增逆向推理，1420/10000 SELF_PROXY；原F恰为共同代理，所以本模型两列相同。

07:53增量：D10k/D50k正向已各完成3600更新，分别07:47:42/07:48:39保存stage权重（da758c85 /5e07ba94），反向接续未完成。配对64环境已修复，v2唯一owner3470666于07:44:43启动，首例DRC通过并在07:46:51真实EMX启动；07:53:58已有2个身份核对通过的fresh结果，2/2均proxy与EM联合未命中，另62未验证，不能算作失败或64总体0%成绩。新快照 ../paired64_20260921_v1/received_snapshots/snapshot_0001/analysis/ 已含真实配对CSV和目标热点图，ANALYSIS_RECEIPT SHA3b75ddfd08cedb3112d3834ecbe0cf0e076fc1c7ba99eba2a73ab9eecbbb9993；8文件传输SHA通过，Lp–Ls图已目视。普通结果接收PID3481284在MARS运行，60s读取固定64终态，源/配置指向v2，只新增内容才出图，不依赖本机SSH持续在线，不控制模拟器。

## 实际训练

08:15:36 UTC只读确认：以下5个训练PID均R，评价3486449、曲线导出3509894、原生3470666、物理接收3481284均存在；没有重新提交。日志和输出目录均为MARS路径：

| 组 | PID | 实际日志（同目录保存权重、配置、终态） |
|---|---:|---|
| D10k |3409830|`/volumes/research-localdata/ywang3652/eucap_paper_dataset_size_811_20260921_v1/runs/D10k/console.log`|
| D50k |3409831|`/volumes/research-localdata/ywang3652/eucap_paper_dataset_size_811_20260921_v1/runs/D50k/console.log`|
| Small |3457206|`/volumes/research-localdata/ywang3652/eucap_paper_architecture_100k_20260921_v1/runs/Small/console.log`|
| Medium |3457207|`/volumes/research-localdata/ywang3652/eucap_paper_architecture_100k_20260921_v1/runs/Medium/console.log`|
| Large |3457208|`/volumes/research-localdata/ywang3652/eucap_paper_architecture_100k_20260921_v1/runs/Large/console.log`|

- 数据量线：D10k总10000=8000/1000/1000，D50k总50000=40000/5000/5000；固定原非对称F/I、seed20260713，从零训练。07:23:53启动，PID3409830/3409831。F各3600已保存，I继续；I更新数UNKNOWN，不以CPU当权重更新计数。
- D100k总100000=78891/9740/11369。原seed20260713合同/分区/normalizer精确匹配，复用原权重；原软件环境UNKNOWN。依据 ../dataset_size_811_20260921_v1/D100K_REUSE.json。
- 宽度线：同一原100K split，各自先F后冻结F训练I；Small128×3、Medium256×3、Large512×3。07:32:41启动，PID3457206/3457207/3457208。07:39 cut已保存F更新1440/720/240（各预算3600）；不是原非对称D100k，也不是6329实验。
- 两队列分别2×4及3×4 BLAS线程、nice19。每模型3600F+3600I；validation选checkpoint，test不参与训练或选模。更新观察不改变优化数学。
- 远端日志根：/volumes/research-localdata/ywang3652/eucap_paper_dataset_size_811_20260921_v1/ 和 /volumes/research-localdata/ywang3652/eucap_paper_architecture_100k_20260921_v1/；具体PID、日志及cut在 TRAINING_OBSERVATION_20260921T073905Z.json。
- 每完成模型的普通评价接续已经实际运行；当前原D100k与新Small评价完成，另四组尚无完整F/I终态，不画缺失数据量趋势。

## 已交P0热点图
../p0_target_hotspots_20260921_v1/：Lp–Ls、Q–|k|各计数及联合命中率，4组CSV/SVG/PNG。
只复用保存T10000与原seed13预测，0新推理；1420/10000=14.20%，SELF_PROXY。20×20目标坐标格，相同范围和0..1色标，Q–K有7个N<10格标线；失败仍在原分母。这些取代旧响应坐标密度图作为主要覆盖表达。
原uniform目标PCG64 seed20260921，与旧centered-LHS seed20260810不是同一个T10k。模型Q是minimum而非精确Q；无Q扫描。

## EM证据分开
- 新uniform10000：已冻结无放回64配对候选。旧v1的Calibre环境FAIL留证；07:44:43唯一v2 owner3470666使用既有mentor/old/2025入口接续，旧owner退役。首2 actual GDS复用，不重复Cadence；07:53:58首2 fresh结果接收。08:05 owner及接收程序仍存在。预算64/原截止2026-09-22T07:18:37.138583Z/8GiB跨v1+v2累计，没有重置。
- 三个advisor历史100K案例保留独立，不替代新T10k验证；旧未配对IoU不用于主结论。
- 已找回历史200k fixed10k中7298个实际EMX身份闭合配对，其中目标K≤0.8为5992。是不同模型/目标实验，旧统计发布NO-GO仍有效；未读取受门禁EM数值，不能给物理命中率。精确来源见 ../dataset_size_811_20260921_v1/historical_experiment_census_20260921_v1/EXPERIMENT_CENSUS.json。
- 历史RQ-I实际phase_FAIL、7/10臂、3/5完整pair，不是正在运行，也不补跑。

## 下一合法入口
保持剩余4个健康训练、唯一原生v2以及已运行的评价、物理接收和曲线导出程序，不重复启动。下一增量为完成模型的曲线/成本导出以及新的冻结配对EM结果；同family全部完成才运行已准备的publication_figures.py（SHAa0dd3cfb405a631a47a0cd69dcc5e059dce0b784188b34ef2ece5e1dd56748f5）。任何新结果必须绑定其模型、normalizer、split、评价及真值类别。私有资料不推公共GitHub。
