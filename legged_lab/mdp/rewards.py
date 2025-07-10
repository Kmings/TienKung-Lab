# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.tienkung.tienkung_env import TienKungEnv


def track_lin_vel_xy_yaw_frame_exp(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv,
    # 标准差，用于计算指数奖励，控制奖励衰减速度
    std: float,
    # 场景实体配置，默认为名为 "robot" 的实体配置
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    计算在偏航坐标系下跟踪 xy 方向线速度的指数奖励。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。
    std (float): 标准差，用于计算指数奖励，控制奖励衰减速度。
    asset_cfg (SceneEntityCfg, 可选): 场景实体配置，默认为名为 "robot" 的实体配置。

    返回:
    torch.Tensor: 指数奖励张量。
    """
    # 从环境场景中获取指定名称的关节物体
    asset: Articulation = env.scene[asset_cfg.name]
    # 将机器人在世界坐标系下的线速度转换到偏航坐标系下
    # math_utils.yaw_quat(asset.data.root_quat_w) 提取根节点四元数的偏航部分
    # math_utils.quat_rotate_inverse 用于将线速度转换到偏航坐标系
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    # 计算期望线速度（命令中的线速度）与偏航坐标系下线速度的误差平方和
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    # 根据误差计算指数奖励，误差越小奖励越接近 1，误差越大奖励越接近 0
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv,
    # 标准差，用于计算指数奖励，控制奖励衰减速度
    std: float,
    # 场景实体配置，默认为名为 "robot" 的实体配置
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    计算在世界坐标系下跟踪 z 轴角速度的指数奖励。
    
    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。
    std (float): 标准差，用于计算指数奖励，控制奖励衰减速度。
    asset_cfg (SceneEntityCfg, 可选): 场景实体配置，默认为名为 "robot" 的实体配置。

    返回:
    torch.Tensor: 指数奖励张量。
    """
    # 从环境场景中获取指定名称的关节物体
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算期望角速度（命令中的角速度）与世界坐标系下 z 轴角速度的误差平方
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    # 根据误差计算指数奖励，误差越小奖励越接近 1，误差越大奖励越接近 0
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv,
    # 场景实体配置，默认为名为 "robot" 的实体配置
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    计算机器人在本体坐标系下 z 轴线速度的平方值。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。
    asset_cfg (SceneEntityCfg, 可选): 场景实体配置，默认为名为 "robot" 的实体配置。

    返回:
    torch.Tensor: 机器人在本体坐标系下 z 轴线速度的平方值张量。
    """
    # 从环境场景中获取指定名称的关节物体
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算并返回机器人在本体坐标系下 z 轴线速度的平方值
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv,
    # 场景实体配置，默认为名为 "robot" 的实体配置
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    计算机器人在本体坐标系下 xy 轴角速度平方和。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。
    asset_cfg (SceneEntityCfg, 可选): 场景实体配置，默认为名为 "robot" 的实体配置。

    返回:
    torch.Tensor: 机器人在本体坐标系下 xy 轴角速度平方和的张量。
    """
    # 从环境场景中获取指定名称的关节物体
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算并返回机器人在本体坐标系下 xy 轴角速度的平方和
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv,
    # 场景实体配置，默认为名为 "robot" 的实体配置
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    计算关节施加扭矩与关节速度乘积的范数。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。
    asset_cfg (SceneEntityCfg, 可选): 场景实体配置，默认为名为 "robot" 的实体配置。

    返回:
    torch.Tensor: 关节施加扭矩与关节速度乘积的范数张量。
    """
    # 从环境场景中获取指定名称的关节物体
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算关节施加扭矩与关节速度绝对值乘积的范数
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv,
    # 场景实体配置，默认为名为 "robot" 的实体配置
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    计算机器人指定关节加速度的平方和。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。
    asset_cfg (SceneEntityCfg, 可选): 场景实体配置，默认为名为 "robot" 的实体配置。

    返回:
    torch.Tensor: 机器人指定关节加速度平方和的张量。
    """
    # 从环境场景中获取指定名称的关节物体
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算并返回机器人指定关节加速度的平方和
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv
) -> torch.Tensor:
    """
    计算相邻两次动作之间差值的平方和。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。

    返回:
    torch.Tensor: 相邻两次动作之间差值平方和的张量。
    """
    # 计算动作缓冲区中最后两个动作之间差值的平方和
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def undesired_contacts(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv,
    # 接触力阈值，用于判断是否为非期望接触
    threshold: float,
    # 传感器配置，用于获取接触力数据
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    判断是否存在超过阈值的非期望接触。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。
    threshold (float): 接触力阈值，用于判断是否为非期望接触。
    sensor_cfg (SceneEntityCfg): 传感器配置，用于获取接触力数据。

    返回:
    torch.Tensor: 表示是否存在非期望接触的布尔张量。
    """
    # 从环境场景中获取指定名称的接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取接触传感器的历史接触力数据
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # 判断是否存在超过阈值的接触力
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv,
    # 接触力阈值，用于判断是否处于飞行状态
    threshold: float,
    # 传感器配置，用于获取接触力数据
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    判断机器人是否处于飞行状态。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。
    threshold (float): 接触力阈值，用于判断是否处于飞行状态。
    sensor_cfg (SceneEntityCfg): 传感器配置，用于获取接触力数据。

    返回:
    torch.Tensor: 表示机器人是否处于飞行状态的布尔张量。
    """
    # 从环境场景中获取指定名称的接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取接触传感器的历史接触力数据
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # 判断是否存在超过阈值的接触力
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # 若接触次数总和小于 0.5，则认为机器人处于飞行状态
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv,
    # 场景实体配置，默认为名为 "robot" 的实体配置
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    计算机器人在本体坐标系下投影重力向量 xy 分量的平方和。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。
    asset_cfg (SceneEntityCfg, 可选): 场景实体配置，默认为名为 "robot" 的实体配置。

    返回:
    torch.Tensor: 机器人在本体坐标系下投影重力向量 xy 分量平方和的张量。
    """
    # 从环境场景中获取指定名称的关节物体
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算并返回投影重力向量 xy 分量的平方和
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv
) -> torch.Tensor:
    """
    对非因情节超时导致终止的情节进行惩罚。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。

    返回:
    torch.Tensor: 表示是否对情节进行惩罚的布尔张量。
    """
    # 返回重置标志与非超时标志的逻辑与结果
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv,
    # 奖励的最大阈值
    threshold: float,
    # 传感器配置，用于获取接触时间数据
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    计算双足机器人单脚支撑时的奖励，奖励与单脚支撑时间相关。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。
    threshold (float): 奖励的最大阈值。
    sensor_cfg (SceneEntityCfg): 传感器配置，用于获取接触时间数据。

    返回:
    torch.Tensor: 双足机器人单脚支撑时的奖励张量。
    """
    # 从环境场景中获取指定名称的接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取双脚当前的腾空时间
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    # 获取双脚当前的接触时间
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    # 判断双脚是否处于接触状态
    in_contact = contact_time > 0.0
    # 根据接触状态选择接触时间或腾空时间
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    # 判断是否处于单脚支撑状态
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    # 计算单脚支撑时的奖励
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    # 对奖励进行裁剪，使其不超过最大阈值
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_slide(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv,
    # 传感器配置，用于获取接触力数据
    sensor_cfg: SceneEntityCfg,
    # 场景实体配置，默认为名为 "robot" 的实体配置
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    计算机器人脚滑动的奖励。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。
    sensor_cfg (SceneEntityCfg): 传感器配置，用于获取接触力数据。
    asset_cfg (SceneEntityCfg, 可选): 场景实体配置，默认为名为 "robot" 的实体配置。

    返回:
    torch.Tensor: 机器人脚滑动的奖励张量。
    """
    # 从环境场景中获取指定名称的接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 判断是否有接触力大于 1.0 的情况
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # 从环境场景中获取指定名称的关节物体
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取指定身体部位在世界坐标系下的线速度
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # 计算脚滑动的奖励
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv,
    # 传感器配置，用于获取接触力数据
    sensor_cfg: SceneEntityCfg,
    # 接触力阈值，默认为 500
    threshold: float = 500,
    # 最大奖励值，默认为 400
    max_reward: float = 400
) -> torch.Tensor:
    """
    计算机器人身体受到的接触力奖励。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。
    sensor_cfg (SceneEntityCfg): 传感器配置，用于获取接触力数据。
    threshold (float, 可选): 接触力阈值，默认为 500。
    max_reward (float, 可选): 最大奖励值，默认为 400。

    返回:
    torch.Tensor: 机器人身体受到的接触力奖励张量。
    """
    # 从环境场景中获取指定名称的接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取接触传感器的法向接触力
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    # 接触力小于阈值时，奖励置为 0
    reward[reward < threshold] = 0
    # 接触力大于阈值时，奖励减去阈值
    reward[reward > threshold] -= threshold
    # 对奖励进行裁剪，使其在 0 到最大奖励值之间
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv,
    # 场景实体配置，默认为名为 "robot" 的实体配置
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    计算机器人关节位置与默认位置偏差的 L1 范数。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。
    asset_cfg (SceneEntityCfg, 可选): 场景实体配置，默认为名为 "robot" 的实体配置。

    返回:
    torch.Tensor: 机器人关节位置与默认位置偏差 L1 范数的张量。
    """
    # 从环境场景中获取指定名称的关节物体
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算关节位置与默认位置的偏差
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # 判断命令速度是否接近 0
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    # 计算关节位置偏差的 L1 范数，并乘以速度标志
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def body_orientation_l2(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv,
    # 场景实体配置，默认为名为 "robot" 的实体配置
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    计算机器人身体朝向与重力向量在 xy 平面偏差的 L2 范数。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。
    asset_cfg (SceneEntityCfg, 可选): 场景实体配置，默认为名为 "robot" 的实体配置。

    返回:
    torch.Tensor: 机器人身体朝向与重力向量在 xy 平面偏差 L2 范数的张量。
    """
    # 从环境场景中获取指定名称的关节物体
    asset: Articulation = env.scene[asset_cfg.name]
    # 将重力向量转换到指定身体部位的坐标系下
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    # 计算身体朝向与重力向量在 xy 平面偏差的 L2 范数
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv,
    # 传感器配置，用于获取接触力数据
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    判断机器人脚是否有绊倒的情况。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。
    sensor_cfg (SceneEntityCfg): 传感器配置，用于获取接触力数据。

    返回:
    torch.Tensor: 表示机器人脚是否有绊倒情况的布尔张量。
    """
    # 从环境场景中获取指定名称的接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 判断是否有脚的水平接触力大于垂直接触力 5 倍的情况
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    # 传入的环境对象，类型可以是 BaseEnv 或者 TienKungEnv
    env: BaseEnv | TienKungEnv,
    # 场景实体配置，默认为名为 "robot" 的实体配置
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # 双脚之间的最小距离阈值，默认为 0.2
    threshold: float = 0.2
) -> torch.Tensor:
    """
    计算人形机器人双脚距离过近的惩罚值。

    参数:
    env (BaseEnv | TienKungEnv): 传入的环境对象。
    asset_cfg (SceneEntityCfg, 可选): 场景实体配置，默认为名为 "robot" 的实体配置。
    threshold (float, 可选): 双脚之间的最小距离阈值，默认为 0.2。

    返回:
    torch.Tensor: 人形机器人双脚距离过近的惩罚值张量。
    """
    # 断言指定的身体部位数量为 2
    assert len(asset_cfg.body_ids) == 2
    # 从环境场景中获取指定名称的关节物体
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取双脚在世界坐标系下的位置
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    # 计算双脚之间的距离
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    # 计算双脚距离过近的惩罚值，最小为 0
    return (threshold - distance).clamp(min=0)


# Regularization Reward
def ankle_torque(
    # 传入的环境对象，类型为 TienKungEnv
    env: TienKungEnv
) -> torch.Tensor:
    """
    对脚踝关节施加的大扭矩进行惩罚。

    参数:
    env (TienKungEnv): 传入的环境对象。

    返回:
    torch.Tensor: 脚踝关节扭矩惩罚值张量。
    """
    # 计算脚踝关节施加扭矩的平方和
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)


def ankle_action(
    # 传入的环境对象，类型为 TienKungEnv
    env: TienKungEnv
) -> torch.Tensor:
    """
    对脚踝关节的动作进行惩罚。

    参数:
    env (TienKungEnv): 传入的环境对象。

    返回:
    torch.Tensor: 脚踝关节动作惩罚值张量。
    """
    # 计算脚踝关节动作的绝对值之和
    return torch.sum(torch.abs(env.action[:, env.ankle_joint_ids]), dim=1)


def hip_roll_action(
    # 传入的环境对象，类型为 TienKungEnv
    env: TienKungEnv
) -> torch.Tensor:
    """
    对髋关节滚动关节的动作进行惩罚。

    参数:
    env (TienKungEnv): 传入的环境对象。

    返回:
    torch.Tensor: 髋关节滚动关节动作惩罚值张量。
    """
    # 计算髋关节滚动关节动作的绝对值之和
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[0], env.right_leg_ids[0]]]), dim=1)


def hip_yaw_action(
    # 传入的环境对象，类型为 TienKungEnv
    env: TienKungEnv
) -> torch.Tensor:
    """
    对髋关节偏航关节的动作进行惩罚。

    参数:
    env (TienKungEnv): 传入的环境对象。

    返回:
    torch.Tensor: 髋关节偏航关节动作惩罚值张量。
    """
    # 计算髋关节偏航关节动作的绝对值之和
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[2], env.right_leg_ids[2]]]), dim=1)


def feet_y_distance(
    # 传入的环境对象，类型为 TienKungEnv
    env: TienKungEnv
) -> torch.Tensor:
    """
    当命令的 y 轴速度较低时，对双脚在 y 轴上的距离进行惩罚，以保持合理间距。

    参数:
    env (TienKungEnv): 传入的环境对象。

    返回:
    torch.Tensor: 双脚在 y 轴上距离的惩罚值张量。
    """
    # 计算左脚相对于根节点的位置
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    # 计算右脚相对于根节点的位置
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    # 将左脚位置转换到根节点坐标系下
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :], leftfoot))
    # 将右脚位置转换到根节点坐标系下
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :], rightfoot))
    # 计算双脚在根节点坐标系下 y 轴的距离与目标距离的差值的绝对值
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - 0.299)
    # 判断命令的 y 轴速度是否较低
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    # 计算双脚在 y 轴上距离的惩罚值
    return y_distance_b * y_vel_flag


# Periodic gait-based reward function
def gait_clock(
    # 归一化的步态相位，取值范围在 [0, 1]
    phase: torch.Tensor,
    # 步态周期中摆动相所占比例
    air_ratio: torch.Tensor,
    # 相位边界的过渡宽度，用于平滑插值
    delta_t: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    生成用于脚摆动相和支撑相的周期性步态时钟信号。

    此函数构建两个与相位相关的信号：
    - `I_frc`：在摆动相期间激活（用于惩罚地面力）
    - `I_spd`：在支撑相期间激活（用于惩罚脚速）

    摆动相和支撑相之间的过渡在 `delta_t` 范围内进行平滑处理，以创建可微分的过渡。

    参数:
    phase (torch.Tensor): 归一化的步态相位，取值范围在 [0, 1]，形状为 [num_envs]。
    air_ratio (torch.Tensor): 步态周期中摆动相所占比例，形状为 [num_envs]。
    delta_t (float): 相位边界的过渡宽度，用于平滑插值。

    返回:
    tuple[torch.Tensor, torch.Tensor]: 
        - I_frc (torch.Tensor): 基于步态的摆动相时钟信号，范围 [0, 1]，形状为 [num_envs]。
        - I_spd (torch.Tensor): 基于步态的支撑相时钟信号，范围 [0, 1]，形状为 [num_envs]。

    注意:
    - 边界处的过渡（例如，摆动相→支撑相）采用线性插值。
    - 用于奖励塑造，将预期行为与步态相位相关联。
    """
    # 判断是否处于摆动相
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    # 判断是否处于支撑相
    stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))

    # 判断是否处于第一个过渡阶段
    trans_flag1 = phase < delta_t
    # 判断是否处于第二个过渡阶段
    trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
    # 判断是否处于第三个过渡阶段
    trans_flag3 = phase > (1 - delta_t)

    # 计算摆动相时钟信号
    I_frc = (
        1.0 * swing_flag
        + (0.5 + phase / (2 * delta_t)) * trans_flag1
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
        + 0.0 * stand_flag
        + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
    )
    # 计算支撑相时钟信号
    I_spd = 1.0 - I_frc
    return I_frc, I_spd


def gait_feet_frc_perio(
    # 传入的环境对象，类型为 TienKungEnv
    env: TienKungEnv,
    # 相位边界的过渡宽度，用于平滑插值，默认为 0.02
    delta_t: float = 0.02
) -> torch.Tensor:
    """
    在步态的摆动相期间惩罚脚受到的力。

    参数:
    env (TienKungEnv): 传入的环境对象。
    delta_t (float, 可选): 相位边界的过渡宽度，用于平滑插值，默认为 0.02。

    返回:
    torch.Tensor: 步态摆动相脚力惩罚值张量。
    """
    # 获取左脚摆动相的掩码
    left_frc_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    # 获取右脚摆动相的掩码
    right_frc_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    # 计算左脚在摆动相的脚力得分
    left_frc_score = left_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    # 计算右脚在摆动相的脚力得分
    right_frc_score = right_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    # 返回左右脚在摆动相的脚力得分之和
    return left_frc_score + right_frc_score


def gait_feet_spd_perio(
    # 传入的环境对象，类型为 TienKungEnv
    env: TienKungEnv,
    # 相位边界的过渡宽度，用于平滑插值，默认为 0.02
    delta_t: float = 0.02
) -> torch.Tensor:
    """
    在步态的支撑相期间惩罚脚的速度。

    参数:
    env (TienKungEnv): 传入的环境对象。
    delta_t (float, 可选): 相位边界的过渡宽度，用于平滑插值，默认为 0.02。

    返回:
    torch.Tensor: 步态支撑相脚速惩罚值张量。
    """
    # 获取左脚支撑相的掩码
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    # 获取右脚支撑相的掩码
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    # 计算左脚在支撑相的脚速得分
    left_spd_score = left_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    # 计算右脚在支撑相的脚速得分
    right_spd_score = right_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    # 返回左右脚在支撑相的脚速得分之和
    return left_spd_score + right_spd_score


def gait_feet_frc_support_perio(
    # 传入的环境对象，类型为 TienKungEnv
    env: TienKungEnv,
    # 相位边界的过渡宽度，用于平滑插值，默认为 0.02
    delta_t: float = 0.02
) -> torch.Tensor:
    """
    奖励在支撑相期间提供适当支撑力的行为。

    参数:
    env (TienKungEnv): 传入的环境对象。
    delta_t (float, 可选): 相位边界的过渡宽度，用于平滑插值，默认为 0.02。

    返回:
    torch.Tensor: 步态支撑相脚支撑力奖励值张量。
    """
    # 获取左脚支撑相的掩码
    left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    # 获取右脚支撑相的掩码
    right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    # 计算左脚在支撑相的脚支撑力得分
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    # 计算右脚在支撑相的脚支撑力得分
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    # 返回左右脚在支撑相的脚支撑力得分之和
    return left_frc_score + right_frc_score
