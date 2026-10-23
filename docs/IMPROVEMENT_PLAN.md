# 核查审阅与修改方案（IMPROVEMENT_PLAN，v2）

日期：2026-10-22。审阅对象：本仓库 `main@d9be385`（下称 **R**），参考实现 `supermimic_vibed@a1508d4`（Isaac Lab 3 kit-less + MuJoCo-Warp，下称 **V**），论文 arXiv:2610.28480v1（19 页 PDF 逐页通读）。仿真底座保持 mjlab `8bb2bc9`。

## 0. 结论摘要

R 的工程主体（数据管线、可微 FK、解析奖励/观测头、状态读写、世界模型、BPTT、critic、回报采样、多卡）实现完整，并且有数值对拍测试。它与论文目标的差距主要来自下面三点，都是解读或协议上的偏差，不是工程缺陷：

1. **策略观测口径的解读（影响最大）。** R 的"论文口径"（e2/e6/e7）把 g_t 解释为**纯参考量**，策略拿不到任何"参考相对于机器人"的信息，因此无法纠正航向漂移：整条 clip 评测中平均 yaw 误差 82°，世界系的四项奖励全部失效，评测回报卡在 0.56。论文写明动作跟踪接口"follows the BeyondMimic formulation; the changes we make are noted inline"，正文标出的改动只有两处：K=10 未来帧，以及 Table A.1 的本体观测。mjlab 注册的 G1 跟踪任务是无状态估计版（`has_state_estimation=False`），它去掉的是 `motion_anchor_pos_b` 和 `base_lin_vel`，**保留** `motion_anchor_ori_b`。这个量可以由 IMU 观测，只与参考有关，属于目标侧。Table A.1 的五项本体观测正好等于 BeyondMimic 无状态估计 actor 观测去掉"命令 + 锚点朝向"两个目标项后的剩余部分。所以更合理的读法是：g_t 包含参考锚点相对机器人锚点的朝向（6D）。V 正是这样做的（观测加 6 维参考根朝向），1.47 h 达到 0.829、0/40 失败。
2. **评测协议 bug。** `eval/benchmark.py` 用"存活步数"做归一化分母。论文 §4 的定义是"undiscounted sum ... over the full episode, divided by the largest value this sum can take. A policy that terminates early therefore scores low"。现在的写法让失败 clip 只按存活部分打分，**高估**了带失败的检查点。例如 e3 @24k 有 5/40 失败，报告的 0.824 偏高。
3. **"迭代"的定义。** Algorithm A.1 写"Both ω and θ update every iteration"，所以 1 次迭代 = 1 次更新，按 UTD 0.4 计含 2.5 个仿真步。R 按读法 (A) 把迭代当成仿真步，导致余弦学习率按更新数计衰减快 2.5 倍，320k 迭代的总训练量也只有论文的 1/2.5，检查点间隔与 Fig. A.1 的"每 4000 次迭代"不对应。V 按更新计。

另外有一处次要偏差：策略目标被除以 H（`loss = -J / H`），使梯度缩小 32 倍，Table A.5 的"全局范数 1.0、逐元素 0.1"裁剪几乎从不触发，与论文的裁剪工况不同。

## 1. 逐条发现（按严重度）

| # | 严重度 | 位置 | 发现 | 证据 | 修改 |
|---|---|---|---|---|---|
| F1 | 高 | `eval/benchmark.py`（`norm_return=ret/(max_r*st)`） | 失败 clip 的归一化分母取存活步数，不是整条 clip | 论文 §4 归一化回报定义；V `evaluate.py` 用 `clip_len-1` | 分母改为整条 clip 应走的步数 L；成功 clip 断言 steps==L；JSON 同时保留旧口径，便于对照历史结果 |
| F2 | 高 | `config.py: anchor_obs: bool=False`；`docs/EXPERIMENTS.md` "正式结果统一论文观测口径" | 纯参考目标无法观测航向误差，策略系统性漂移 | `diag_drift`：yaw 误差 82°，去漂移后速度奖励 0.83（与锚点策略的 0.87 相当）；mjlab G1 无状态估计任务保留 `motion_anchor_ori_b`；V 加 6D 相对朝向后 0/40 失败 | `anchor_obs` 改为三值 `none / ori / pos_ori`，**默认 `ori`**（6 维，IMU 可观测，部署不需要状态估计）；`pos_ori` 等价于原 e3；`none` 等价于原 e2。旧检查点里的 bool 自动映射 |
| F3 | 中 | `training/loop.py`、`ScheduleCfg` | 迭代按仿真步计；学习率衰减、总长度、检查点间隔都与 A.1/A.5 不一致 | Algorithm A.1 "Both ω and θ update every iteration"；Table A.5 "Policy updates per world-model update 1" | 迭代 = 1 次 WM 更新（10 批）+ 1 次策略/critic 更新；每次迭代采集 1/UTD = 2.5 个仿真步（小数累加）；320k 迭代，前 250k 余弦衰减到 0.03×；每 4000 迭代存检查点；日志记录仿真步数 |
| F4 | 中 | `imagination/policy_trainer.py: loss = -J / cfg.horizon` | 目标被除以 H，裁剪工况与论文不同 | Eq. 2 无 1/H；V 用 `-J.mean()` | 默认 `loss = -J`（batch 均值），保留 `objective_scale="per_step"` 选项 |
| F5 | 中 | `policy/nets.py: Critic` | critic 只有规范化状态 + 目标 + ξ + 4 维根误差，要从 71+680 维原始量里自己"算"出跟踪误差；(256,128) 的小网络很难做到，H 步之外的自举因此偏粗 | V 的 critic 显式输入 14 个刚体的位置误差（航向系）、关节误差、根误差和相对朝向 | critic 增加特权跟踪误差特征：锚点相对位姿 9 + 14 刚体位置误差 42 + 关节误差 29（连同原 4 维根误差共 84 维）。只进 critic，不进策略。V_ψ(ŝ_H) 对 ŝ_H 的梯度经 FK 保留 |
| F6 | 低 | `motion_library.goal_features` | 每帧 68 维只有本帧的局部量，未来帧之间的相对位姿（往哪转、往哪走）要靠策略自己积分角速度 | V 的每帧目标含"未来参考根相对当前参考根"的位移 3 + 旋转 6D | 新增可选目标表示 `goal_features="traj"`：每帧 77 维 = 原 68 维 + 相对当前目标帧航向系的根位移 3 + 根旋转 6D。默认仍为 `v1`，用实验决定是否切换 |
| F7 | 低 | `training/loop.py` | 首次更新就同时训练策略，此时世界模型还是零初始化的匀速外推，前几百次策略梯度没有信息量 | V：500 次纯 WM 预热 | 加 `wm_warmup_iterations`（默认 500，约 1–2 min），计入偏离项 |
| F8 | 低 | 全部 MLP | 全程 TF32。5090 上 bf16 autocast 明显更快 | V 实测提速 24% | 加 `amp="none"/"bf16"`，默认 `none`（与论文的 TF32 一致），作为提速选项做对照 |
| F9 | 待核 | `tracking_env_cfg.py: nconmax=35, njmax=250` + `warn_overflow=0` | 溢出提示被静默后，看不到摔倒/躺地动作是否丢接触或约束 | e1b 记录"早期策略每步触发"；V 在 Newton 下需要 njmax=500/nconmax=128 | 先读取 mujoco_warp 的溢出标志做计数（TensorBoard 记录），确有溢出再提高容量 |

复核通过、保持不变的部分：Eq. 1（8 步自回归、状态 + BCE）、Eq. 2/3（软存活权重、critic 自举并保留对 ŝ_H 的梯度、GAE 只在想象段上计算、目标固定）、Table A.2 奖励（逐项与 mjlab 对拍）、Table A.3 DR（摩擦/encoder_bias 按回合采样、COM 在 startup 采样）、终止阈值（骨盆高度 0.25、倾斜 0.8）、回报采样（50/40/10，G_max=10）、K=10、历史 10 帧、网络尺寸、超参、多卡 lr×√N、BONES-SEED 过滤（129,943 条，论文 129,514）。

## 2. 修改清单（文件级）

| 文件 | 改动 |
|---|---|
| `config.py` | `EnvCfg.anchor_obs: str = "ori"`；`EnvCfg.goal_features: str = "v1"`；`ScheduleCfg` 改为按更新计的迭代（`total_iterations=320_000`、`lr_decay_iterations=250_000`、`checkpoint_interval=4000`、`warmup_env_steps=40`、`wm_warmup_iterations=500`）；顶层 `amp` |
| `envs/tracking_env_cfg.py`、`envs/mdp/observations.py` | goal 组按 `anchor_obs` 追加 `anchor_ori`（6）或 `anchor_pos + anchor_ori`（9）；`goal_features="traj"` 时用新特征 |
| `envs/mdp/motion_library.py` | `goal_window(..., mode)` 支持 `traj`（相对当前目标帧的位姿） |
| `world_model/heads.py` | `anchor_relative_obs` 按模式切片；新增 `critic_error_features(s, ref, bodies)` |
| `imagination/compiled.py`、`rollout.py` | 每步返回 critic 误差特征；目标拼接按 `anchor_obs` 模式；`ImagRollout` 增加 `crit_feat` |
| `imagination/policy_trainer.py` | `loss=-J`（可选 per_step）；critic 用新特征；bf16 autocast |
| `policy/nets.py` | `Critic` 输入维度 = 71 + goal + 33 + 84；`PolicyCfg.goal_dim` 由配置推导；可选 autocast |
| `training/loop.py` | 迭代 = 更新；采集 2.5 步/迭代；WM 预热；记录 `env_steps`；检查点保存 schedule 状态 |
| `eval/benchmark.py`、`scripts/*.py` | F1 修正；`anchor_obs`/`goal_features` 从检查点配置读取（兼容旧 bool）；JSON 同时输出新旧两种口径 |
| `tests/` | 新增/更新：评测分母、`anchor_obs` 三种模式下仿真观测与解析观测逐元素一致、critic 特征形状与梯度、迭代计数 |
| `docs/EXPERIMENTS.md`、`README.md` | 记录 v2 决定与结果；结果表注明评测口径 |

兼容性：旧检查点的 `anchor_obs: true/false` 自动映射为 `pos_ori/none`；旧检查点的 critic 维度不同，只支持评测，不支持续训。

## 3. 验证与实验计划（本机 8×RTX 5090，GPU 0 已被他人占用，只用 1–7）

前置：`uv sync`，下载 LAFAN1 G1 CSV（40 条）→ 50 Hz npz（核对总长 2.45 h），全部单元测试通过。

| 运行 | 配置 | 目的 |
|---|---|---|
| v2-ori | 新默认（anchor 朝向 + F3/F4/F5/F7） | 主结果：论文 Fig. 4a 的 LAFAN1 单卡对照 |
| v2-ori-traj | + `goal_features=traj` | F6 是否有收益 |
| v2-none | `anchor_obs=none` | 在新评测口径下复核"纯参考目标"的上限 |
| v2-posori | `anchor_obs=pos_ori` | 需要状态估计时的上界（对应 e3） |
| v2-ori-oldcritic | 关闭 F5 | critic 特征的消融 |
| v2-ori-bf16 | `amp=bf16` | 提速及其对精度的影响 |

每个运行单卡 3 h 上限，每 4000 次迭代存检查点，检查点在空闲卡上离线评测：40 条整 clip，确定性，摩擦 1.0，无噪声无 DR，使用修正后的分母。

**验收标准**：v2-ori 或 v2-ori-traj 在单张 5090 上 ≤2 h 内评测回报 ≥0.82（修正口径），失败 ≤2/40；世界模型 8 步误差稳定；全部测试通过。达标后，把同一配置用于 BONES-SEED 全量（需另外下载 23.5 GB、转换、过滤、打包），对照 Table A.8。

## 4. 风险

- `ori` 读法仍是对论文的解释，论文没有逐字列出 g_t 的内容。文档中把 `none / ori / pos_ori` 三种结果并列报告，不隐藏任何一种。
- F3 改变了迭代的含义，旧日志中的"迭代"不能和新日志直接对比；新日志同时记录 `env_steps` 和墙钟时间。
- 5090 与 H100 的算力不同，"1.2 h 到 0.82"只能按墙钟时间做量级对照。

## 5. v3：采纳 supermimic_vibed 中此前未采纳的部分（2026-10-22，依据 v2 跑满 3 h 的结果）

v2 跑满 3 h 后，所有 `ori` 变体都在 0.79–0.81、1/40 失败，距论文 0.82 只差一点，而且在这台 CPU 偏慢的机器上每次迭代要 0.8 s。因此第二轮把 vibed 中此前没有采纳的部分全部补上，分为"提速"和"算法选项"两类。

| # | vibed 的做法 | 本项目的实现 | 配置项（v3 默认） |
|---|---|---|---|
| V1 | 想象循环每步只做策略 + 世界模型 + 观测，奖励/终止/critic 特征在 rollout 结束后批量计算 | `imagination/compiled.py`：`ImagStep`（锚点只算骨盆→躯干短链 FK）+ `ImagHeads`（批量全身 FK）；与旧实现逐元素一致 | 始终启用 |
| V2 | 世界模型的 k 步窗口整体编译 | `WorldModel._window` 一个编译图；10 个批次共用掩码；fused Adam；采样不做 host 同步 | 始终启用 |
| V3 | 世界模型输入/输出用滑动均值方差归一化 + LayerNorm | `RunningNorm`（原地更新、无同步、2e8 样本后冻结）；输出为一步增量，`StateSpec.target_delta` 是积分的精确逆 | `world_model.normalization=running`、`layer_norm=True` |
| V4 | 解析软终止（sigmoid 软化阈值） | `world_model.termination=analytic`，宽度 0.02 m / 0.04 | `analytic` |
| V5 | 策略输入滑动归一化、ELU | `policy.input_norm=running`、`activation=elu`；统计量来自仿真观测，随检查点保存 | 是 |
| V6 | 想象起点离 clip 末尾至少 H 帧 | `policy_trainer.start_margin=32` | 32 |
| V7 | 关闭自碰撞 | `env.self_collision=False`（mjlab `FULL_COLLISION_WITHOUT_SELF`） | False |
| V8 | 目标为相对当前参考帧的轨迹特征 | `env.goal_features=traj` | traj |

仍未采纳：以骨盆计算根部奖励、vibed 的重置采样记账方式。bf16 起初未采纳（旧的逐步 rollout 下反而更慢），批量 heads 之后实测快 31%，2026-10-23 起设为默认（见 EXPERIMENTS.md）。

兼容性：旧检查点的配置里没有 `input_norm` / `activation` / `self_collision`，评测脚本自动按旧架构和旧物理（`fixed` / `silu` / 开启自碰撞）重建。

验证：见 `docs/EXPERIMENTS.md` 的 v3 / v4 小节（3 个种子 + 3 组消融：打开自碰撞、世界模型恢复固定归一化、策略恢复固定归一化）。

## 6. 最终结果与配置决定（2026-10-22）

v4（v3 默认配置）在 LAFAN1 上单张 RTX 5090 训练 3 h，3 个种子分别为 0.843 / 0.843 / 0.848，**全部 0/40 失败**；1.6 h（12000 次迭代）时三种子平均 0.809。论文 Fig. 4a 为单张 H100 1.2 h 达到 0.82。与论文的时间差距主要来自本机宿主 CPU（训练受 kernel launch 限制）；按迭代数衡量，与 vibed 达到 0.82 所需的约 12000 次迭代一致。

消融结论（单种子）：策略输入归一化和关闭自碰撞主要加快前期学习，3 h 时差别消失；世界模型的滑动归一化 / LayerNorm / 解析终止没有可测的影响。默认配置保持 v3（唯一做过 3 个种子验证的配置），世界模型选项记为"中性"；需要与 BeyondMimic 物理一致时使用 `--env.self-collision`。

全部数字见 `docs/EXPERIMENTS.md` 的 v4 小节；用 `scripts/summarize_runs.py` 可重新生成表格。
