# UAV-MEC 仿真与可视化脚本片段

本仓库保存无人机辅助移动边缘计算研究中的一个环境仿真与绘图入口：`main.py`。

**当前公开内容是脚本片段，尚不能独立运行，也不是完整的 DDQN/PPO 训练工程。**

## 脚本做什么

- 构造手工资源策略：均分时间模板、最大功率档位、固定流态天线端口。
- 构造手工轨迹策略：每个时隙朝目标终点移动。
- 调用环境运行一个 episode，打印状态和逐时隙信息。
- 绘制奖励、信息年龄（AoI）、任务积压、安全速率、任务服务量、能耗和无人机轨迹。

## 文件与依赖状态

| 项目 | 公开状态 |
| :--- | :--- |
| `main.py` | 已包含 |
| `action_space.py` / `ActionSpace` | 当前仓库未包含 |
| `config.py` / `get_default_config` | 当前仓库未包含 |
| `uav_mec_env.py` / `UAVMECEnv` | 当前仓库未包含 |
| DDQN/PPO 训练代码、模型权重与实验配置 | 当前仓库未包含 |

脚本使用 Python、NumPy 和 Matplotlib。由于上述自定义模块尚未公开，仅安装第三方包仍不足以运行。本仓库目前不提供完整复现命令或可验证的性能提升结论。

## 所属研究

研究方向是流态天线赋能的无人机辅助移动边缘计算安全传输，关注端口选择、资源分配、任务卸载与轨迹优化。

[查看作者主页](https://github.com/ye056343-collab) · [联系邮箱](mailto:yepinyi@zjut.edu.cn)

---

## English

This repository contains `main.py`, a simulation and plotting excerpt from a UAV-assisted mobile edge computing project. It uses hand-crafted resource and trajectory policies and visualizes rewards, AoI, backlog, secrecy rate, service amounts, energy, and UAV trajectories.

The imported `action_space.py`, `config.py`, and `uav_mec_env.py` modules are not included. DDQN/PPO training code, checkpoints, and experiment configurations are also absent. This is not a standalone runnable project or a complete reproduction package.
