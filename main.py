from action_space import ActionSpace
import math
import numpy as np

from config import get_default_config
from uav_mec_env import UAVMECEnv


def build_simple_resource_action(env: UAVMECEnv, action_space: ActionSpace):
    """
    一个最简单的手工资源策略：
    1. 时间模板采用均分
    2. 每个用户都用最大功率档位
    3. 所有用户/GJ/UAV-S 的 FAS 端口都先固定为 0
    """
    time_template_idx = action_space.get_equal_share_template_index()

    # 每个用户一个功率档位索引
    power_level_idx_list = action_space.get_default_power_index_list(
        action_space.get_max_power_index()
    )

    # 每个用户一个端口索引
    user_port_idx_list = action_space.get_default_user_port_index_list(0)

    # GJ / UAV-S 端口先固定 0
    gj_port_idx = 0
    uav_port_idx = 0

    resource_action = action_space.build_env_action(
        time_template_idx=time_template_idx,
        power_level_idx_list=power_level_idx_list,
        user_port_idx_list=user_port_idx_list,
        gj_port_idx=gj_port_idx,
        uav_port_idx=uav_port_idx,
    )
    return resource_action


def build_simple_trajectory_action(env: UAVMECEnv):
    """
    一个最简单的手工轨迹策略：
    UAV 每个时隙朝终点方向飞
    """
    current_pos = env.uav_position
    target_pos = np.array(env.scenario.uav_s_end, dtype=float)

    direction = target_pos - current_pos
    norm_dir = np.linalg.norm(direction)

    if norm_dir > 1e-10:
        move_angle = math.atan2(direction[1], direction[0])
    else:
        move_angle = 0.0

    move_distance = min(5.0, env.cfg.uav_speed_max * env.cfg.delta_t)

    traj_action = {
        "move_angle": float(move_angle),
        "move_distance": float(move_distance),
    }
    return traj_action


def build_simple_action(env: UAVMECEnv, action_space: ActionSpace):
    """
    兼容 env.step(action) 的总动作：
    = 资源动作 + 轨迹动作
    """
    resource_action = build_simple_resource_action(env, action_space)
    traj_action = build_simple_trajectory_action(env)

    action = {**resource_action, **traj_action}
    return action


def run_episode():
    cfg = get_default_config()
    env = UAVMECEnv(cfg)
    action_space = ActionSpace(cfg)

    state = env.reset()
    print("========== 初始状态 ==========")
    print("current_slot =", state["current_slot"])
    print("uav_position =", state["uav_position"])
    print("uav_energy   =", state["uav_energy"])
    print("backlog      =", state["backlog"])
    print("aoi          =", state["aoi"])
    print("================================")

    done = False
    step_count = 0

    while not done:
        action = build_simple_action(env, action_space)
        next_state, reward, done, info = env.step(action)
        env.print_step_info(info, reward)
        state = next_state
        step_count += 1

    print("\n========== 最终状态 ==========")
    print("总步数        =", step_count)
    print("最终时隙      =", state["current_slot"])
    print("最终 UAV 位置 =", np.round(state["uav_position"], 4))
    print("最终 UAV 能量 =", round(state["uav_energy"], 4))
    print("最终 backlog  =", np.round(state["backlog"], 4))
    print("最终 AoI      =", np.round(state["aoi"], 4))
    print("================================")

    return env


def plot_results(env: UAVMECEnv):
    """
    画结果图
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n未安装 matplotlib，跳过绘图。")
        return

    # 1. 总奖励曲线
    plt.figure()
    plt.plot(env.history["reward_total"])
    plt.xlabel("Slot")
    plt.ylabel("Reward Total")
    plt.title("Total Reward per Slot")
    plt.grid(True)

    # 2. 资源层奖励曲线
    plt.figure()
    plt.plot(env.history["reward_resource"])
    plt.xlabel("Slot")
    plt.ylabel("Reward Resource")
    plt.title("Resource Reward per Slot")
    plt.grid(True)

    # 3. 轨迹层奖励曲线
    plt.figure()
    plt.plot(env.history["reward_traj"])
    plt.xlabel("Slot")
    plt.ylabel("Reward Trajectory")
    plt.title("Trajectory Reward per Slot")
    plt.grid(True)

    # 4. 平均 AoI 曲线
    plt.figure()
    plt.plot(env.history["avg_aoi"])
    plt.xlabel("Slot")
    plt.ylabel("Average AoI")
    plt.title("Average AoI per Slot")
    plt.grid(True)

    # 5. 平均 backlog 曲线
    plt.figure()
    plt.plot(env.history["avg_backlog"])
    plt.xlabel("Slot")
    plt.ylabel("Average Backlog")
    plt.title("Average Backlog per Slot")
    plt.grid(True)

    # 6. UAV 轨迹图
    plt.figure()
    plt.plot(env.history["uav_x"], env.history["uav_y"], marker="o")
    plt.scatter(
        [env.scenario.uav_s_start[0]], [env.scenario.uav_s_start[1]],
        marker="s", s=80, label="UAV Start"
    )
    plt.scatter(
        [env.scenario.uav_s_end[0]], [env.scenario.uav_s_end[1]],
        marker="*", s=120, label="UAV End"
    )

    user_x = [p[0] for p in env.scenario.user_positions]
    user_y = [p[1] for p in env.scenario.user_positions]
    plt.scatter(user_x, user_y, marker="^", s=80, label="Users")

    plt.scatter(
        [env.scenario.gj_position[0]], [env.scenario.gj_position[1]],
        marker="x", s=100, label="GJ"
    )

    # 把 UAV-E 轨迹也画出来，便于看几何关系
    plt.plot(
        env.scenario.uav_e_trajectory[:, 0],
        env.scenario.uav_e_trajectory[:, 1],
        linestyle="--",
        label="UAV-E Trajectory"
    )

    xmin, xmax, ymin, ymax = env.cfg.service_region
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("UAV Trajectory")
    plt.legend()
    plt.grid(True)

    # 7. 平均安全速率曲线
    plt.figure()
    plt.plot(env.history["avg_secure_rate"])
    plt.xlabel("Slot")
    plt.ylabel("Average Secure Rate")
    plt.title("Average Secure Rate per Slot")
    plt.grid(True)

    # 8. 服务量分解曲线
    plt.figure()
    plt.plot(env.history["avg_local_done"], label="Average Local Done")
    plt.plot(env.history["avg_offload_done"], label="Average Offload Done")
    plt.plot(env.history["avg_total_done"], label="Average Total Done")
    plt.xlabel("Slot")
    plt.ylabel("Average Service Amount")
    plt.title("Service Amount Decomposition per Slot")
    plt.legend()
    plt.grid(True)

    # 9. 平均本地能耗曲线
    plt.figure()
    plt.plot(env.history["avg_local_energy"])
    plt.xlabel("Slot")
    plt.ylabel("Average Local Energy")
    plt.title("Average Local Energy per Slot")
    plt.grid(True)

    # 10. 平均发射能耗曲线
    plt.figure()
    plt.plot(env.history["avg_tx_energy"])
    plt.xlabel("Slot")
    plt.ylabel("Average TX Energy")
    plt.title("Average TX Energy per Slot")
    plt.grid(True)

    # 11. 平均 UAV 计算能耗曲线
    plt.figure()
    plt.plot(env.history["avg_uav_compute_energy"])
    plt.xlabel("Slot")
    plt.ylabel("Average UAV Compute Energy")
    plt.title("Average UAV Compute Energy per Slot")
    plt.grid(True)

    # 12. 平均用户总能耗曲线
    plt.figure()
    plt.plot(env.history["avg_total_user_energy"])
    plt.xlabel("Slot")
    plt.ylabel("Average Total User Energy")
    plt.title("Average Total User Energy per Slot")
    plt.grid(True)

    # 13. 资源层总能耗曲线
    plt.figure()
    plt.plot(env.history["resource_energy_total"])
    plt.xlabel("Slot")
    plt.ylabel("Resource Energy Total")
    plt.title("Resource Energy per Slot")
    plt.grid(True)

    # 14. 系统总能耗曲线
    plt.figure()
    plt.plot(env.history["total_system_energy"])
    plt.xlabel("Slot")
    plt.ylabel("Total System Energy")
    plt.title("Total System Energy per Slot")
    plt.grid(True)

    # 15. 用户总能耗曲线
    plt.figure()
    plt.plot(env.history["sum_total_user_energy"])
    plt.xlabel("Slot")
    plt.ylabel("Total User Energy")
    plt.title("Total User Energy per Slot")
    plt.grid(True)

    # 16. 飞行能耗曲线
    plt.figure()
    plt.plot(env.history["flight_energy"])
    plt.xlabel("Slot")
    plt.ylabel("Flight Energy")
    plt.title("Flight Energy per Slot")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    env = run_episode()
    plot_results(env)