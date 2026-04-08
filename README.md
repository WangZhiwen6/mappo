# MAPPO-LSTM for Multi-Zone HVAC Control

本项目实现了一个面向建筑多区域空调控制的 `CTDE + MAPPO + LSTM` 训练框架。
当前场景中，`56` 个区域/末端作为 `56` 个智能体进行协同控制。

## 核心设定（工程视角）

- 多智能体数量：`N = 56`
- 动作空间：联合连续动作 `a_t \in R^56`（每个区域一个控制量）
- 观测组织：
  - `local_obs`：每个智能体的局部观测（用于 actor）
  - `global_state`：整栋楼全局状态（用于 centralized critic）
- 算法范式：`CTDE`（训练中心化，执行去中心化）
- 奖励：单一**全局标量奖励**（所有智能体共享），由 `Myreward` 计算

## 交互阶段与更新阶段：Actor/Critic 如何运行

### 1) 环境交互阶段（rollout）

在每个时间步 `t`，执行顺序如下：

1. 策略前向：`actor + critic` 同时运行  
   - actor 读取 `local_obs` 输出联合动作  
   - critic 读取 `global_state` 输出 `V(s_t)`
2. 与环境交互：执行 `env.step(a_t)`，得到 `(obs_{t+1}, r_t, done, info)`
3. 写入经验池：存储 `obs_t, a_t, r_t, V(s_t), log_prob(a_t), LSTM states`

说明：
- 每个时间步都会产生**新动作**与**新奖励**。
- 奖励是**联合奖励（全局奖励）**，不是按区域分开的 `56` 个奖励。

相关实现：
- rollout 主循环：`myrppo/mappo_recurrent.py` 的 `collect_rollouts()`
- 环境 step 与奖励计算：`mygym/envs/eplus_env.py` 的 `step()`

### 2) 训练更新阶段（train）

更新阶段不和环境交互，只使用 rollout buffer 里的轨迹：

1. 用当前参数重新计算 `log_prob_new` 和 `V_new`
2. 基于 `log_prob_old` 计算 PPO ratio 与 clipped policy loss
3. 基于 `returns` 和 `V_new` 计算 value loss
4. 加入 entropy 项后反向传播，联合更新 actor/critic 参数

相关实现：
- `myrppo/mappo_recurrent.py` 的 `train()`

## 奖励机制（当前实现）

训练入口使用 `Myreward`：
- 入口配置：`myrppo/rppo.py` 中 `reward_class = Myreward`
- 函数定义：`mygym/utils/rewards.py` 中 `class Myreward`

奖励是单个标量：

\[
r_t = r_t^{energy} + r_t^{temp} + r_t^{co2}
\]

\[
r_t = \lambda_E w_E P_t^{energy} + \lambda_T w_T P_t^{temp} + \lambda_C w_C P_t^{co2}
\]

其中：
- `energy` 项反映 HVAC 能耗惩罚
- `temp` 项反映温度舒适性偏差
- `co2` 项反映空气质量偏差

虽然奖励是单标量，`info` 中会返回分解项（用于日志和诊断），例如：
- `energy_term`
- `comfort_term`
- `co2_term`
- `total_power_demand`
- `total_temperature_violation`
- `total_co2_violation`

## 为什么是联合奖励，而不是每区域单独奖励

这是有意的协同设计，适配建筑系统强耦合特性：

- HVAC 的能耗与舒适性是全局耦合目标
- 单区域最优不等于整楼最优
- 在 CTDE 中，shared global reward + centralized critic 通常更稳定

因此当前实现是“**分布式动作 + 共享全局回报**”的 cooperative MARL 结构。

## 当前网络数量说明

- actor LSTM 网络：`1` 个（参数共享）
- critic LSTM 网络：`1` 个（中心化）
- 每个智能体各自维护自己的 actor hidden/cell state

也就是说：不是 `56` 套 actor-LSTM 参数，而是 `1` 套共享参数 + `56` 份时序记忆状态。
